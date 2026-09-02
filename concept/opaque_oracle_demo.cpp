// ============================================================================
// minlhc — Pedagogical demo: the opaque-oracle boundary and adjoint operators.
// ----------------------------------------------------------------------------
//
// What's new here, on top of tiny_differentiable_pipeline.cpp:
//
// A visible code boundary between an OPAQUE stochastic simulator ("pretend
// GEANT") and a DIFFERENTIABLE downstream. The shower parameter μ (lateral
// width — a Molière-radius / tracker-material analog) lives INSIDE the
// oracle. We do not analytically differentiate it. Instead we execute the
// G4.md thesis in code:
//
//   1. Extract J[cell] = ∂⟨E_cell⟩/∂μ from N controlled oracle runs with
//      matched random seeds (central difference at μ₀).
//   2. Freeze J. From now on, μ-gradients come from composing J with the
//      analytic downstream:
//          dL/dμ = Σ_i  (dL/dA_i) · g · J[i]
//   3. Jointly optimize (g, μ) using this adjoint chain. μ lives inside
//      an opaque simulator, yet it appears in the gradient like any other
//      calibration parameter.
//   4. Validate against a brute-force scan that reruns the full oracle at
//      each μ — the expensive baseline the cheap adjoint must match.
//
// Four built-in diagnostics, each with a CSV export and a pass/fail line:
//
//   T1  FD vs adjoint gradients for g and μ (trust).
//   T2  Extracted J matches direct finite-difference of ⟨E_cell⟩(μ)
//       for a specific cell over a large independent ensemble.
//   T3  Adjoint L(μ) vs brute-force L(μ), and the linearity-window plot
//       that shows WHERE a single J extraction stops being adequate.
//   T4  Joint (g,μ) optimizer recovers injected truth; classical 1D scan
//       over g with μ held at nominal lands at a biased point (the
//       "preconception trap" for an opaque nuisance).
//   T6  Mean-level score control. The per-event score used above is a
//       mean-target pseudo-likelihood, Σ_e (r_e − T)² = N[(r̄ − T)² + Var r];
//       its Var term rewards variance shrinkage and displaces the optimum
//       from truth (measured by a 50-toy ensemble). T6 scans an objective
//       built on the ensemble MEAN — the Var term absent by construction —
//       and checks that its optimum returns to the injected truth,
//       localizing the T4 offset in the score, not the operator machinery.
//
// An oracle-call ledger is printed at the end of the run so the cost
// accounting (extractions, fit forwards, falsifier share) is a measured
// output of the program rather than a hand derivation.
//
// Deliberate pedagogical simplifications:
//   - noise σ = 0 everywhere (orthogonal to the thesis, easy to add back).
//   - matched seeds across μ perturbations for clean per-event FD checks.
//   - J extracted from a separate Particle "probe" so the data ensemble
//     and the Jacobian ensemble are genuinely independent.
//   - Reconstruction is a hard central 3×3 window per layer. Cells that
//     move in/out of the window as μ changes are captured to first order
//     by J — that's exactly the thing a cluster-based E_rec exposes.
//
// Build:   clang++ -std=c++17 -O3 concept/opaque_oracle_demo.cpp \
//                 -o opaque_oracle_demo
// Run:     ./opaque_oracle_demo
// Plot:    python3 concept/plot_oracle_demo.py
//
// \license Apache 2.0
// \copyright 2026 Vassil Vassilev
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace minlhc {

// ---------------------------------------------------------------------------
// Event data model
// ---------------------------------------------------------------------------

struct DetectorGeometry {
  int nLayers;
  int nX;
  int nY;
  double cellSize;
};

struct Calibration {
  double gain = 1.0;        ///< electronics gain g
  double mu = 1.0;          ///< lateral shower width (inside oracle)
  double noiseSigma = 0.0;  ///< electronics noise (off throughout this demo)
};

struct Particle { double px, py, pz, E; };
struct CaloHit  { int layer; double x, y, energy; };
struct Digi     { int cellId; double adc; double trueEnergy; };

/// Cell id = layer * nX * nY + ix * nY + iy, with the geometry centred on
/// the origin so hits near zero are not piled into a clamped edge cell.
inline int cell_id_for(const CaloHit& h, const DetectorGeometry& g) {
  double halfX = 0.5 * g.nX * g.cellSize;
  double halfY = 0.5 * g.nY * g.cellSize;
  int ix = int(std::floor((h.x + halfX) / g.cellSize));
  int iy = int(std::floor((h.y + halfY) / g.cellSize));
  ix = std::max(0, std::min(ix, g.nX - 1));
  iy = std::max(0, std::min(iy, g.nY - 1));
  return h.layer * g.nX * g.nY + ix * g.nY + iy;
}

/// Decode a cell id back into (layer, ix, iy) for window tests.
inline void decode_cell(int cellId, const DetectorGeometry& g,
                        int& layer, int& ix, int& iy) {
  int per_layer = g.nX * g.nY;
  layer = cellId / per_layer;
  int in_layer = cellId % per_layer;
  ix = in_layer / g.nY;
  iy = in_layer % g.nY;
}

// ---------------------------------------------------------------------------
// Oracle-call ledger (instrumentation)
// ---------------------------------------------------------------------------

/// Incremented once per simulate_shower call. A namespace-scope mutable is a
/// deliberate exception to the usual no-globals rule: threading a ledger
/// through every signature would obscure the physics this single-file,
/// single-threaded teaching demo exists to show. It never touches any RNG
/// path, so all physics numbers are unchanged by its presence.
inline uint64_t g_oracle_calls = 0;

// ---------------------------------------------------------------------------
// The opaque oracle (stochastic, treated as a black box)
// ---------------------------------------------------------------------------

/// Stochastic electromagnetic-shower surrogate. μ scales the transverse
/// hit positions — the central trick for matched-seed differencing: the
/// draws are standard normals, so pinning the seed makes (μ ± ε) reuse
/// identical (u₁, u₂), and the central difference of ⟨E_cell⟩ captures
/// only the μ dependence of the binning.
inline std::vector<CaloHit> simulate_shower(const Particle& p, double mu,
                                            std::mt19937& rng,
                                            const DetectorGeometry& geom,
                                            int hitsPerLayer) {
  ++g_oracle_calls;
  std::normal_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> fluct(0.9, 1.1);
  std::vector<CaloHit> hits;
  hits.reserve(geom.nLayers * hitsPerLayer);
  for (int l = 0; l < geom.nLayers; ++l) {
    double e_layer_total = p.E * std::exp(-0.3 * double(l)) * fluct(rng);
    double e_per_hit = e_layer_total / double(hitsPerLayer);
    for (int h = 0; h < hitsPerLayer; ++h) {
      double u1 = unit(rng);
      double u2 = unit(rng);
      hits.push_back({l, mu * u1, mu * u2, e_per_hit});
    }
  }
  return hits;
}

// ---------------------------------------------------------------------------
// Differentiable downstream
// ---------------------------------------------------------------------------

inline std::vector<Digi> digitize(const std::vector<CaloHit>& hits,
                                  const Calibration& calib,
                                  const DetectorGeometry& geom,
                                  std::mt19937& rng) {
  std::map<int, double> cellE;
  for (const auto& h : hits) cellE[cell_id_for(h, geom)] += h.energy;

  const bool use_noise = (calib.noiseSigma > 0.0);
  std::normal_distribution<double> noise(0.0,
      use_noise ? calib.noiseSigma : 1.0);

  std::vector<Digi> digis;
  digis.reserve(cellE.size());
  for (const auto& kv : cellE) {
    double n = use_noise ? noise(rng) : 0.0;
    digis.push_back({kv.first, calib.gain * kv.second + n, kv.second});
  }
  return digis;
}

/// Classify a cell by its position relative to the shower centre.
///   Core:   max(|ix-cx|, |iy-cy|) <= r_core          (central (2r+1)×(2r+1))
///   Halo:   r_core < max(|ix-cx|, |iy-cy|) <= r_halo (the annulus)
///   Else:   outside the fiducial region              (ignored)
///
/// The two regions form two *independent* observables. Their ratio depends
/// on μ but not on g — which is precisely what breaks the (g, μ) degeneracy
/// a single-scalar E_rec would leave.
inline int cell_region(int cellId, const DetectorGeometry& g,
                       int r_core, int r_halo) {
  int layer, ix, iy;
  decode_cell(cellId, g, layer, ix, iy);
  int cx = g.nX / 2, cy = g.nY / 2;
  int m = std::max(std::abs(ix - cx), std::abs(iy - cy));
  if (m <= r_core) return 0;   // core
  if (m <= r_halo) return 1;   // halo
  return 2;                    // outside fiducial
}

struct RecoPair { double E_core; double E_halo; };

inline RecoPair reconstruct(const std::vector<Digi>& digis,
                            const DetectorGeometry& g,
                            int r_core, int r_halo) {
  RecoPair out{0.0, 0.0};
  for (const auto& d : digis) {
    int reg = cell_region(d.cellId, g, r_core, r_halo);
    if (reg == 0) out.E_core += d.adc;
    else if (reg == 1) out.E_halo += d.adc;
  }
  return out;
}

/// Gaussian log-likelihood on BOTH observables. Core and halo are combined
/// as independent Gaussians with the same σ — a deliberate pedagogical
/// simplification (real shower-shape fits use a more structured covariance).
inline double log_lik_pair(const RecoPair& r, const RecoPair& target,
                           double sigma) {
  double rc = (r.E_core - target.E_core) / sigma;
  double rh = (r.E_halo - target.E_halo) / sigma;
  return -0.5 * (rc * rc + rh * rh);
}

// ---------------------------------------------------------------------------
// Response Jacobian — the G4.md thesis in code
// ---------------------------------------------------------------------------

/// J[cell] = ∂⟨E_cell⟩/∂μ at μ₀, estimated from matched-seed central
/// differences over nSeeds oracle runs. Cells inside the central region
/// have J < 0 (lose energy as μ grows); cells just outside have J > 0
/// (gain energy). The adjoint sees exactly this picture.
struct ResponseJacobian {
  std::map<int, double> dE_dmu;
  double mu0 = 0.0;
  int nSeeds = 0;
  double epsilon = 0.0;
};

inline ResponseJacobian extract_jacobian(const Particle& probe,
                                         const DetectorGeometry& geom,
                                         int hitsPerLayer,
                                         double mu0, double epsilon,
                                         int nSeeds,
                                         uint64_t base_seed = 0xC0FFEEu) {
  ResponseJacobian J;
  J.mu0 = mu0;
  J.nSeeds = nSeeds;
  J.epsilon = epsilon;

  std::map<int, double> sum;
  for (int s = 0; s < nSeeds; ++s) {
    std::mt19937 rng_p(base_seed + uint64_t(s));
    std::mt19937 rng_m(base_seed + uint64_t(s));
    auto hp = simulate_shower(probe, mu0 + epsilon, rng_p, geom, hitsPerLayer);
    auto hm = simulate_shower(probe, mu0 - epsilon, rng_m, geom, hitsPerLayer);
    std::map<int, double> Ep, Em;
    for (const auto& h : hp) Ep[cell_id_for(h, geom)] += h.energy;
    for (const auto& h : hm) Em[cell_id_for(h, geom)] += h.energy;
    for (const auto& kv : Ep) sum[kv.first] += kv.second;
    for (const auto& kv : Em) sum[kv.first] -= kv.second;
  }
  for (auto& kv : sum)
    J.dE_dmu[kv.first] = kv.second / (2.0 * epsilon * double(nSeeds));
  return J;
}

// ---------------------------------------------------------------------------
// Adjoint composition
// ---------------------------------------------------------------------------

struct EventGrads { double dg = 0.0, dmu = 0.0, logL = 0.0; };

/// Per-event gradient contribution through two observables (core + halo).
/// The oracle is never reinvoked; μ-gradients come from the frozen J.
///
///   dL/dg  = Σ_{obs}  dL/dE_obs · Σ_{i ∈ obs}  E_cell_i
///   dL/dμ  = Σ_{obs}  dL/dE_obs · Σ_{i ∈ obs}  g · J[cell_i]
///
/// Shadow-cell accounting: J[cell] can be non-zero at cells with no energy
/// in the current event's digis (those cells would acquire energy if μ
/// perturbed). Their contribution to dL/dμ is real — they are lit in the
/// ensemble mean. We include them explicitly.
inline EventGrads event_grads(const std::vector<Digi>& digis,
                              const Calibration& calib,
                              const DetectorGeometry& geom,
                              int r_core, int r_halo,
                              const ResponseJacobian& J,
                              const RecoPair& target, double sigma) {
  EventGrads out;
  RecoPair r = reconstruct(digis, geom, r_core, r_halo);
  out.logL = log_lik_pair(r, target, sigma);

  double dL_dEc = -(r.E_core - target.E_core) / (sigma * sigma);
  double dL_dEh = -(r.E_halo - target.E_halo) / (sigma * sigma);

  std::map<int, double> present;  // cellId -> trueEnergy for nominal digis
  for (const auto& d : digis) present[d.cellId] = d.trueEnergy;

  // Sweep all cells in the response Jacobian's support.
  for (const auto& kv : J.dE_dmu) {
    int reg = cell_region(kv.first, geom, r_core, r_halo);
    if (reg == 2) continue;
    double dL_dE = (reg == 0) ? dL_dEc : dL_dEh;
    auto it = present.find(kv.first);
    double trueE = (it != present.end()) ? it->second : 0.0;
    out.dg  += dL_dE * trueE;
    out.dmu += dL_dE * calib.gain * kv.second;
    if (it != present.end()) present.erase(it);
  }
  // Cells in digis but NOT in J still contribute to dg via their trueEnergy.
  for (const auto& kv : present) {
    int reg = cell_region(kv.first, geom, r_core, r_halo);
    if (reg == 2) continue;
    double dL_dE = (reg == 0) ? dL_dEc : dL_dEh;
    out.dg += dL_dE * kv.second;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Whole-pipeline forwards
// ---------------------------------------------------------------------------

/// Reruns the opaque oracle at calib.mu — expensive baseline.
inline double forward_logL(const std::vector<Particle>& particles,
                           const DetectorGeometry& geom,
                           const Calibration& calib,
                           int r_core, int r_halo, int hitsPerLayer,
                           const RecoPair& target, double sigma,
                           uint64_t seed = 42) {
  std::mt19937 rng(seed);
  double total = 0.0;
  for (const auto& p : particles) {
    auto hits = simulate_shower(p, calib.mu, rng, geom, hitsPerLayer);
    auto digs = digitize(hits, calib, geom, rng);
    RecoPair r = reconstruct(digs, geom, r_core, r_halo);
    total += log_lik_pair(r, target, sigma);
  }
  return total;
}

/// Forward using a linearised oracle output: nominal digis are carried along
/// and each cell's true energy is updated by ΔE_cell = J[cell] · dμ; the
/// ADC is updated consistently (Δadc = g · ΔE_cell). All downstream
/// non-linearity (region gating, two-observable likelihood) is applied
/// exactly — only the oracle is treated as a first-order operator.
inline double linearised_logL(const std::vector<std::vector<Digi>>& nom,
                              const Calibration& calib,
                              const DetectorGeometry& geom,
                              int r_core, int r_halo,
                              const ResponseJacobian& J, double dmu,
                              const RecoPair& target, double sigma) {
  double total = 0.0;
  for (const auto& dv : nom) {
    // Build per-cell ADC starting from nominal, then apply J·dμ everywhere.
    // Cells present in J but not in the event acquire the mean-expected
    // adjoint increment (shadow cells). Noise is held fixed (σ=0 in this
    // demo; any non-zero noise stays at its nominal draw).
    std::map<int, Digi> d_by_cell;
    for (const auto& d : dv) d_by_cell[d.cellId] = d;
    for (const auto& kv : J.dE_dmu) {
      double dE = kv.second * dmu;
      auto it = d_by_cell.find(kv.first);
      if (it == d_by_cell.end()) {
        d_by_cell[kv.first] = {kv.first, calib.gain * dE, dE};
      } else {
        it->second.trueEnergy += dE;
        it->second.adc        += calib.gain * dE;
      }
    }
    std::vector<Digi> d2;
    d2.reserve(d_by_cell.size());
    for (auto& kv : d_by_cell) d2.push_back(kv.second);
    RecoPair r = reconstruct(d2, geom, r_core, r_halo);
    total += log_lik_pair(r, target, sigma);
  }
  return total;
}

// ---------------------------------------------------------------------------
// Dataset helpers
// ---------------------------------------------------------------------------

struct Dataset {
  std::vector<Particle> particles;
  std::vector<std::vector<Digi>> nominal_digis;
  RecoPair target{0.0, 0.0};
  Calibration truth;
};

inline Dataset make_dataset(int N, const DetectorGeometry& geom,
                            const Calibration& truth,
                            int r_core, int r_halo, int hitsPerLayer,
                            uint64_t seed = 2024) {
  Dataset ds;
  ds.truth = truth;
  ds.particles.reserve(N);
  Particle truth_p{0.0, 0.0, 0.0, 45.0};
  for (int i = 0; i < N; ++i) ds.particles.push_back(truth_p);

  std::mt19937 rng(seed);
  double sum_c = 0.0, sum_h = 0.0;
  int good = 0;
  for (const auto& p : ds.particles) {
    auto hits = simulate_shower(p, truth.mu, rng, geom, hitsPerLayer);
    auto digs = digitize(hits, truth, geom, rng);
    RecoPair r = reconstruct(digs, geom, r_core, r_halo);
    if (r.E_core > 0.0) { sum_c += r.E_core; sum_h += r.E_halo; ++good; }
  }
  ds.target.E_core = (good > 0) ? sum_c / double(good) : 1.0;
  ds.target.E_halo = (good > 0) ? sum_h / double(good) : 1.0;
  return ds;
}

inline void cache_nominal_digis(Dataset& ds, const DetectorGeometry& geom,
                                const Calibration& at_nominal,
                                int hitsPerLayer, uint64_t seed = 42) {
  ds.nominal_digis.clear();
  ds.nominal_digis.reserve(ds.particles.size());
  std::mt19937 rng(seed);
  for (const auto& p : ds.particles) {
    auto hits = simulate_shower(p, at_nominal.mu, rng, geom, hitsPerLayer);
    auto digs = digitize(hits, at_nominal, geom, rng);
    ds.nominal_digis.push_back(std::move(digs));
  }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

struct TestResult {
  std::string name;
  bool ok;
  double value;
  double tol;
  std::string note;
};

inline void print_result(const TestResult& r) {
  std::printf("  [%s] %-52s  value=%-12.4g  tol=%-8.2g  %s\n",
              r.ok ? "OK " : "!! ", r.name.c_str(), r.value, r.tol,
              r.note.c_str());
}

/// T1a: FD vs adjoint for dL/dg. Analytic downstream => FD-precise agreement.
inline TestResult t1_fd_vs_adjoint_g(const Dataset& ds,
                                     const DetectorGeometry& geom,
                                     int r_core, int r_halo, int hitsPerLayer,
                                     const Calibration& at,
                                     const ResponseJacobian& J, double sigma,
                                     uint64_t seed) {
  const double eps = 1e-4;
  Calibration cm = at, cp = at;
  cm.gain -= eps; cp.gain += eps;
  double Lm = forward_logL(ds.particles, geom, cm, r_core, r_halo,
                           hitsPerLayer, ds.target, sigma, seed);
  double Lp = forward_logL(ds.particles, geom, cp, r_core, r_halo,
                           hitsPerLayer, ds.target, sigma, seed);
  double fd = (Lp - Lm) / (2.0 * eps);

  double adj = 0.0;
  std::mt19937 rng(seed);
  for (const auto& p : ds.particles) {
    auto hits = simulate_shower(p, at.mu, rng, geom, hitsPerLayer);
    auto digs = digitize(hits, at, geom, rng);
    auto g = event_grads(digs, at, geom, r_core, r_halo, J, ds.target, sigma);
    adj += g.dg;
  }
  double rel = std::fabs(fd - adj) / (std::fabs(fd) + 1e-12);
  return {"T1a FD vs adjoint dL/dg (fully analytic path)",
          rel < 1e-4, rel, 1e-4, ""};
}

/// T1b: FD vs adjoint for dL/dμ. Adjoint uses frozen J; FD reruns the oracle
/// with matched seeds. Agreement is statistical (depends on J's ensemble).
/// ε_FD is matched to J.epsilon so FD and J measure the slope over the
/// same μ window — otherwise a wider ε_J captures a secant and a narrow
/// ε_FD captures a tangent, and they disagree by construction.
inline TestResult t1_fd_vs_adjoint_mu(const Dataset& ds,
                                      const DetectorGeometry& geom,
                                      int r_core, int r_halo, int hitsPerLayer,
                                      const Calibration& at,
                                      const ResponseJacobian& J, double sigma,
                                      uint64_t seed) {
  const double eps = J.epsilon;  // matched ε
  Calibration cm = at, cp = at;
  cm.mu -= eps; cp.mu += eps;
  double Lm = forward_logL(ds.particles, geom, cm, r_core, r_halo,
                           hitsPerLayer, ds.target, sigma, seed);
  double Lp = forward_logL(ds.particles, geom, cp, r_core, r_halo,
                           hitsPerLayer, ds.target, sigma, seed);
  double fd = (Lp - Lm) / (2.0 * eps);

  double adj = 0.0;
  std::mt19937 rng(seed);
  for (const auto& p : ds.particles) {
    auto hits = simulate_shower(p, at.mu, rng, geom, hitsPerLayer);
    auto digs = digitize(hits, at, geom, rng);
    auto g = event_grads(digs, at, geom, r_core, r_halo, J, ds.target, sigma);
    adj += g.dmu;
  }
  double rel = std::fabs(fd - adj) / (std::fabs(fd) + 1e-12);
  return {"T1b FD vs adjoint dL/dμ (through frozen J)",
          rel < 0.25, rel, 0.25, "statistical; shrinks as nSeeds↑"};
}

/// T2: Adjoint-prediction stability. Extract a SECOND J from an independent
/// stream; both J's are noisy per-cell, but what we actually use is their
/// adjoint prediction L_adj(μ₀ + Δμ). That downstream quantity averages
/// per-cell noise and should agree across independent J extractions within
/// the linearity window.
inline TestResult t2_two_J_predictions_agree(const Dataset& ds,
                                             const DetectorGeometry& geom,
                                             int r_core, int r_halo,
                                             int hitsPerLayer,
                                             const Calibration& base_nominal,
                                             const ResponseJacobian& J_A,
                                             double sigma, double dmu_test,
                                             uint64_t base_seed) {
  ResponseJacobian J_B = extract_jacobian({0,0,0,45}, geom, hitsPerLayer,
                                           J_A.mu0, J_A.epsilon, J_A.nSeeds,
                                           base_seed + 0xBADD);
  double L_A = linearised_logL(ds.nominal_digis, base_nominal, geom,
                               r_core, r_halo, J_A, dmu_test,
                               ds.target, sigma);
  double L_B = linearised_logL(ds.nominal_digis, base_nominal, geom,
                               r_core, r_halo, J_B, dmu_test,
                               ds.target, sigma);
  double rel = std::fabs(L_A - L_B) / (std::fabs(L_A) + 1e-12);
  char note[96];
  std::snprintf(note, sizeof(note),
                "two independent J's, nSeeds=%d each, Δμ=%.3f",
                J_A.nSeeds, dmu_test);
  return {"T2 Two independent J's agree on L_adj(μ₀+Δμ)",
          rel < 0.01, rel, 0.01, note};
}

/// T3 builder: return (μ_i, L_brute_i, L_adjoint_i) over a μ sweep.
struct MuScan { std::vector<double> mu, L_brute, L_adjoint; };

inline MuScan scan_mu(const Dataset& ds, const DetectorGeometry& geom,
                      int r_core, int r_halo, int hitsPerLayer,
                      const Calibration& base_nominal,
                      const ResponseJacobian& J, double sigma,
                      double mu_min, double mu_max, int nPoints,
                      uint64_t seed) {
  MuScan s;
  s.mu.reserve(nPoints);
  s.L_brute.reserve(nPoints);
  s.L_adjoint.reserve(nPoints);
  for (int i = 0; i < nPoints; ++i) {
    double mu = mu_min + (mu_max - mu_min) * double(i) / double(nPoints - 1);
    Calibration c = base_nominal;
    c.mu = mu;
    double Lb = forward_logL(ds.particles, geom, c, r_core, r_halo,
                             hitsPerLayer, ds.target, sigma, seed);
    double La = linearised_logL(ds.nominal_digis, base_nominal, geom,
                                r_core, r_halo, J, mu - J.mu0,
                                ds.target, sigma);
    s.mu.push_back(mu);
    s.L_brute.push_back(Lb);
    s.L_adjoint.push_back(La);
  }
  return s;
}

/// T3: linearity-window width. Starting from μ₀, walk outward in Δμ and
/// find the largest |Δμ| where |L_adj − L_brute| / |L_brute| stays under
/// `tol_curve`. A wide window means the single J extraction is useful for
/// non-local optimization; a narrow one warns that periodic re-extraction
/// is required.
inline TestResult t3_linearity_window(const MuScan& s, double mu0,
                                      double tol_curve = 0.01,
                                      double tol_window = 0.05) {
  int i0 = 0; double d0 = std::fabs(s.mu[0] - mu0);
  for (size_t i = 1; i < s.mu.size(); ++i) {
    double d = std::fabs(s.mu[i] - mu0);
    if (d < d0) { d0 = d; i0 = int(i); }
  }
  int lo = i0, hi = i0;
  auto rel = [&](int i) {
    return std::fabs(s.L_adjoint[i] - s.L_brute[i]) /
           (std::fabs(s.L_brute[i]) + 1e-12);
  };
  while (lo > 0 && rel(lo - 1) < tol_curve) --lo;
  while (hi < int(s.mu.size()) - 1 && rel(hi + 1) < tol_curve) ++hi;
  double width = std::max(std::fabs(s.mu[hi] - mu0),
                          std::fabs(s.mu[lo] - mu0));

  char note[96];
  std::snprintf(note, sizeof(note),
                "μ∈[%.3f, %.3f] passes %.0f%% agreement",
                s.mu[lo], s.mu[hi], tol_curve * 100.0);
  return {"T3 Linearity window (where adjoint L(μ) ≈ brute)",
          width > tol_window, width, tol_window, note};
}

// ---------------------------------------------------------------------------
// Classical 1D scan over g (μ held at the — wrong — nominal)
// ---------------------------------------------------------------------------

struct ScanResult { double best_g; double best_logL; };

inline ScanResult classical_scan_g(const std::vector<Particle>& particles,
                                   const DetectorGeometry& geom,
                                   int r_core, int r_halo, int hitsPerLayer,
                                   double mu_fixed, const RecoPair& target,
                                   double sigma,
                                   double gmin, double gmax, double gstep,
                                   std::vector<double>& gs_out,
                                   std::vector<double>& Ls_out,
                                   uint64_t seed = 42) {
  double best_g = gmin;
  double best_L = -std::numeric_limits<double>::infinity();
  gs_out.clear(); Ls_out.clear();
  for (double g = gmin; g <= gmax + 1e-12; g += gstep) {
    Calibration c; c.gain = g; c.mu = mu_fixed; c.noiseSigma = 0.0;
    double L = forward_logL(particles, geom, c, r_core, r_halo, hitsPerLayer,
                            target, sigma, seed);
    gs_out.push_back(g); Ls_out.push_back(L);
    if (L > best_L) { best_L = L; best_g = g; }
  }
  return {best_g, best_L};
}

// ---------------------------------------------------------------------------
// T6: mean-level score control
// ---------------------------------------------------------------------------

/// Optima of two objectives along one calibration axis through truth,
/// computed from the SAME simulated events per scan point:
///   per-event score  Σ_e log_lik_pair(r_e, T)   (the score used everywhere
///                    above; carries the variance-penalty term), and
///   mean-level score −[(r̄_c − T_c)² + (r̄_h − T_h)²]  (variance term absent
///                    by construction).
struct AxisOptima { double best_mean; double best_evt; };

inline AxisOptima t6_axis_scan(const Dataset& ds, const DetectorGeometry& geom,
                               int r_core, int r_halo, int hitsPerLayer,
                               const Calibration& truth, bool scan_gain,
                               double lo, double hi, double step,
                               double sigma, uint64_t seed) {
  AxisOptima out{lo, lo};
  double Lm = -std::numeric_limits<double>::infinity();
  double Le = -std::numeric_limits<double>::infinity();
  for (double v = lo; v <= hi + 1e-12; v += step) {
    Calibration c = truth;
    (scan_gain ? c.gain : c.mu) = v;
    std::mt19937 rng(seed);
    double evt = 0.0, sum_c = 0.0, sum_h = 0.0;
    for (const auto& p : ds.particles) {
      auto hits = simulate_shower(p, c.mu, rng, geom, hitsPerLayer);
      auto digs = digitize(hits, c, geom, rng);
      RecoPair r = reconstruct(digs, geom, r_core, r_halo);
      evt += log_lik_pair(r, ds.target, sigma);
      sum_c += r.E_core;
      sum_h += r.E_halo;
    }
    double n = double(ds.particles.size());
    double mc = sum_c / n - ds.target.E_core;
    double mh = sum_h / n - ds.target.E_halo;
    double mean_score = -(mc * mc + mh * mh);
    if (mean_score > Lm) { Lm = mean_score; out.best_mean = v; }
    if (evt > Le) { Le = evt; out.best_evt = v; }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Joint optimizer (Adam on (g, μ)) with optional periodic J refresh
// ---------------------------------------------------------------------------

struct OptHistory {
  std::vector<int> iter;
  std::vector<double> g, mu, logL;
  std::vector<int> jacobian_refreshes;  ///< iter values where J was refreshed
};

inline OptHistory run_joint_optimizer(const Dataset& ds,
                                      const DetectorGeometry& geom,
                                      int r_core, int r_halo, int hitsPerLayer,
                                      Calibration start, double sigma,
                                      int nIters,
                                      const Particle& probe,
                                      double j_epsilon, int j_nSeeds,
                                      int j_refresh_every,
                                      uint64_t seed = 42) {
  Calibration cur = start;
  ResponseJacobian J = extract_jacobian(probe, geom, hitsPerLayer, cur.mu,
                                        j_epsilon, j_nSeeds);
  OptHistory hist;
  hist.jacobian_refreshes.push_back(0);

  const double beta1 = 0.9, beta2 = 0.999, eps = 1e-12;
  const double lr_g0 = 0.01, lr_mu0 = 0.01;
  double m_g = 0, v_g = 0, m_u = 0, v_u = 0;

  for (int it = 1; it <= nIters; ++it) {
    if (j_refresh_every > 0 && it > 1 && (it % j_refresh_every) == 0) {
      J = extract_jacobian(probe, geom, hitsPerLayer, cur.mu, j_epsilon,
                           j_nSeeds);
      hist.jacobian_refreshes.push_back(it);
    }
    // Gentle learning-rate decay so the optimizer settles instead of
    // oscillating once it's near the minimum.
    double decay = 1.0 / (1.0 + 0.002 * double(it));
    double lr_g  = lr_g0  * decay;
    double lr_mu = lr_mu0 * decay;

    std::mt19937 rng(seed);
    double tot_dg = 0.0, tot_du = 0.0, tot_L = 0.0;
    for (const auto& p : ds.particles) {
      auto hits = simulate_shower(p, cur.mu, rng, geom, hitsPerLayer);
      auto digs = digitize(hits, cur, geom, rng);
      auto g = event_grads(digs, cur, geom, r_core, r_halo, J,
                           ds.target, sigma);
      tot_dg += g.dg; tot_du += g.dmu; tot_L += g.logL;
    }
    double n = double(ds.particles.size());
    double gg = tot_dg / n;
    double gu = tot_du / n;

    auto adam = [&](double grad, double& m, double& v, double lr, int t) {
      m = beta1 * m + (1.0 - beta1) * grad;
      v = beta2 * v + (1.0 - beta2) * grad * grad;
      double mh = m / (1.0 - std::pow(beta1, double(t)));
      double vh = v / (1.0 - std::pow(beta2, double(t)));
      return lr * mh / (std::sqrt(vh) + eps);
    };
    cur.gain += adam(gg, m_g, v_g, lr_g, it);
    cur.mu   += adam(gu, m_u, v_u, lr_mu, it);

    cur.gain = std::max(0.1, std::min(5.0, cur.gain));
    cur.mu   = std::max(0.1, std::min(5.0, cur.mu));

    hist.iter.push_back(it);
    hist.g.push_back(cur.gain);
    hist.mu.push_back(cur.mu);
    hist.logL.push_back(tot_L);
  }
  return hist;
}

// ---------------------------------------------------------------------------
// CSV writers
// ---------------------------------------------------------------------------

inline void write_csv_header(std::ofstream& f, const char* hdr) { f << hdr << "\n"; }

inline void write_mu_scan_csv(const std::string& path, const MuScan& s) {
  std::ofstream f(path);
  f << std::setprecision(10);
  write_csv_header(f, "mu,L_brute,L_adjoint");
  for (size_t i = 0; i < s.mu.size(); ++i)
    f << s.mu[i] << "," << s.L_brute[i] << "," << s.L_adjoint[i] << "\n";
}

inline void write_gscan_csv(const std::string& path,
                            const std::vector<double>& gs,
                            const std::vector<double>& Ls) {
  std::ofstream f(path);
  f << std::setprecision(10);
  write_csv_header(f, "g,logL");
  for (size_t i = 0; i < gs.size(); ++i) f << gs[i] << "," << Ls[i] << "\n";
}

inline void write_opt_path_csv(const std::string& path, const OptHistory& h) {
  std::ofstream f(path);
  f << std::setprecision(10);
  write_csv_header(f, "iter,g,mu,logL,jacobian_refresh");
  int k = 0;
  for (size_t i = 0; i < h.iter.size(); ++i) {
    int is_refresh = 0;
    while (k < int(h.jacobian_refreshes.size()) &&
           h.jacobian_refreshes[k] < h.iter[i]) ++k;
    if (k < int(h.jacobian_refreshes.size()) &&
        h.jacobian_refreshes[k] == h.iter[i]) is_refresh = 1;
    f << h.iter[i] << "," << h.g[i] << "," << h.mu[i] << "," << h.logL[i]
      << "," << is_refresh << "\n";
  }
}

inline void write_summary_csv(const std::string& path,
                              const Calibration& truth,
                              const Calibration& nominal,
                              const RecoPair& target,
                              double classical_best_g, double classical_best_L,
                              const Calibration& final_joint,
                              double final_joint_L) {
  std::ofstream f(path);
  f << std::setprecision(10);
  write_csv_header(f, "param,truth,nominal_start,classical,joint");
  f << "g,"  << truth.gain << "," << nominal.gain << ","
    << classical_best_g << "," << final_joint.gain << "\n";
  f << "mu," << truth.mu   << "," << nominal.mu   << ","
    << nominal.mu          << "," << final_joint.mu   << "\n";
  f << "logL," << 0.0 << "," << 0.0 << ","
    << classical_best_L << "," << final_joint_L << "\n";
  f << "E_core_target," << target.E_core << ","
    << target.E_core << "," << target.E_core << "," << target.E_core << "\n";
  f << "E_halo_target," << target.E_halo << ","
    << target.E_halo << "," << target.E_halo << "," << target.E_halo << "\n";
}

// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

inline int run() {
  // ----- setup ------------------------------------------------------------
  DetectorGeometry geom{5, 10, 10, 1.0};
  const int hitsPerLayer = 3;
  const int r_core = 1;   // central 3×3 per layer
  const int r_halo = 2;   // halo = 5×5 minus core
  // σ matched to the dominant per-event fluctuation — cell-containment
  // (roughly sqrt(n_hits · p · (1-p)) · ⟨e_hit⟩ ≈ 15 GeV).  Much smaller
  // σ would make the likelihood peak chase stochastic bumps; much larger
  // would flatten it and lose resolution.
  const double sigma = 10.0;

  // Truth has a slightly NARROWER shower than nominal (more energy stays in
  // the core) and a correct gain. The nominal starting point over-estimates
  // μ and under-estimates g — the ridge a 1D g-scan at fixed μ cannot escape.
  Calibration truth;    truth.gain   = 1.00; truth.mu   = 0.85;
  Calibration nominal;  nominal.gain = 0.85; nominal.mu = 1.00;
  Particle probe{0.0, 0.0, 0.0, 45.0};

  const int N = 2000;

  // Oracle-call ledger: phase-boundary snapshots of g_oracle_calls, printed
  // at the end so the cost accounting is a measured output.
  std::vector<std::pair<const char*, uint64_t>> ledger;
  uint64_t last_calls = g_oracle_calls;
  auto mark = [&](const char* label) {
    ledger.push_back({label, g_oracle_calls - last_calls});
    last_calls = g_oracle_calls;
  };

  Dataset ds = make_dataset(N, geom, truth, r_core, r_halo, hitsPerLayer,
                            /*seed=*/2024);
  cache_nominal_digis(ds, geom, nominal, hitsPerLayer, /*seed=*/42);
  mark("dataset generation + cache");

  std::printf("\n==== Opaque-oracle demo =========================================\n");
  std::printf("Truth:   g=%.3f  μ=%.3f\n", truth.gain, truth.mu);
  std::printf("Nominal: g=%.3f  μ=%.3f\n", nominal.gain, nominal.mu);
  std::printf("Dataset: N=%d events, %d hits/layer, core %dx%d + halo %dx%d, σ=%.1f\n",
              N, hitsPerLayer, 2*r_core+1, 2*r_core+1,
              2*r_halo+1, 2*r_halo+1, sigma);
  std::printf("Targets at truth: E_core=%.4f  E_halo=%.4f\n",
              ds.target.E_core, ds.target.E_halo);

  // ----- extract response Jacobian at nominal -----------------------------
  // ε=0.02 gives enough bin-boundary-crossing hits per seed that per-cell
  // estimates are stable; nSeeds=1000 drives the aggregate noise down.
  const double j_eps = 0.02;
  const int j_nSeeds = 1000;
  ResponseJacobian J = extract_jacobian(probe, geom, hitsPerLayer, nominal.mu,
                                        j_eps, j_nSeeds);
  std::printf("Extracted J at μ₀=%.3f using %d matched-seed oracle runs "
              "(ε=%.0e, %zu non-zero cells)\n",
              J.mu0, J.nSeeds, j_eps, J.dE_dmu.size());
  mark("initial J extraction (T1/T2 input)");

  // ----- tests ------------------------------------------------------------
  std::printf("\n---- Diagnostics -----------------------------------------------\n");
  int fails = 0;
  auto t1a = t1_fd_vs_adjoint_g(ds, geom, r_core, r_halo, hitsPerLayer,
                                nominal, J, sigma, /*seed=*/42);
  print_result(t1a); fails += !t1a.ok;
  mark("T1a FD + adjoint pass");

  auto t1b = t1_fd_vs_adjoint_mu(ds, geom, r_core, r_halo, hitsPerLayer,
                                 nominal, J, sigma, /*seed=*/42);
  print_result(t1b); fails += !t1b.ok;
  mark("T1b FD + adjoint pass");

  auto t2 = t2_two_J_predictions_agree(ds, geom, r_core, r_halo,
                                       hitsPerLayer, nominal, J,
                                       sigma, /*dmu_test=*/0.02,
                                       /*base_seed=*/0xC0FFEEu + 1);
  print_result(t2); fails += !t2.ok;
  mark("T2 second extraction (predictions: 0 calls)");

  // Scan μ around nominal for T3 and the linearity-window plot
  MuScan mu_s = scan_mu(ds, geom, r_core, r_halo, hitsPerLayer, nominal,
                        J, sigma, /*mu_min=*/0.60, /*mu_max=*/1.40,
                        /*nPoints=*/41, /*seed=*/42);
  auto t3 = t3_linearity_window(mu_s, nominal.mu,
                                /*tol_curve=*/0.05, /*tol_window=*/0.05);
  print_result(t3); fails += !t3.ok;

  write_mu_scan_csv("mu_scan.csv", mu_s);
  mark("T3 brute-force μ-scan (linearized side: 0)");

  // ----- classical 1D g-scan at nominal μ (the "preconception trap") ------
  std::vector<double> gs_scan, Ls_scan;
  auto classical = classical_scan_g(ds.particles, geom, r_core, r_halo,
                                    hitsPerLayer, nominal.mu, ds.target,
                                    sigma, /*gmin=*/0.70, /*gmax=*/1.40,
                                    /*gstep=*/0.01, gs_scan, Ls_scan);
  write_gscan_csv("gscan_nominal_mu.csv", gs_scan, Ls_scan);
  std::printf("\nClassical 1D g-scan (μ held at %.3f): best_g=%.4f  L=%.3f\n",
              nominal.mu, classical.best_g, classical.best_logL);
  mark("T5 classical g-scan");

  // ----- joint (g, μ) optimizer -------------------------------------------
  const int nIters = 800;
  OptHistory h = run_joint_optimizer(ds, geom, r_core, r_halo, hitsPerLayer,
                                     nominal, sigma, nIters, probe,
                                     j_eps, j_nSeeds,
                                     /*j_refresh_every=*/50, /*seed=*/42);
  Calibration final_joint;
  final_joint.gain = h.g.back();
  final_joint.mu   = h.mu.back();
  final_joint.noiseSigma = 0.0;
  double final_joint_L = h.logL.back();
  write_opt_path_csv("opt_path.csv", h);

  std::printf("Joint optimizer (Adam, %d iters, J refreshed every 50):\n"
              "  final g=%.4f  μ=%.4f  L=%.3f\n",
              nIters, final_joint.gain, final_joint.mu, final_joint_L);
  mark("joint fit (17 J extractions + 800 forwards)");

  // Diagnostic: brute-force L at truth and a nearby grid of the optimizer
  // endpoint lets us see whether we've converged to a local minimum or
  // simply undershoot. If L_at_truth > L_at_optimizer, more work is left.
  double L_truth = forward_logL(ds.particles, geom, truth, r_core, r_halo,
                                hitsPerLayer, ds.target, sigma, /*seed=*/42);
  std::printf("  diagnostic:  L_brute at truth = %.3f (optimum should be ≥ this)\n",
              L_truth);
  mark("truth diagnostic");

  // ----- T4: truth recovery -----------------------------------------------
  double err_g  = std::fabs(final_joint.gain - truth.gain) / truth.gain;
  double err_mu = std::fabs(final_joint.mu   - truth.mu  ) / truth.mu;
  // Tolerance respects the stochastic data-realization noise floor:
  // with finite N, the best-fit point is O(1/√N) from injected truth.
  TestResult t4{"T4 Joint optimizer recovers injected truth",
                err_g < 0.08 && err_mu < 0.08,
                std::max(err_g, err_mu), 0.08, "stochastic floor ~1/√N"};
  print_result(t4); fails += !t4.ok;

  // Classical bias check: we expect the 1D scan to miss truth by > 1%.
  double cbias = std::fabs(classical.best_g - truth.gain) / truth.gain;
  TestResult t5{"T5 Classical 1D scan is biased (|Δg|/g > 1%)",
                cbias > 0.01, cbias, 0.01, "lower bound (bias expected)"};
  print_result(t5); fails += !t5.ok;

  // ----- T6: mean-level score control -------------------------------------
  // Same simulated events per scan point feed both objectives; only the
  // scoring differs. The per-event score's optima sit off truth (its Var
  // term rewards variance shrinkage); the mean-level score's optima must
  // return to truth within the grid step, localizing the T4 offset in the
  // score function.
  AxisOptima ax_g = t6_axis_scan(ds, geom, r_core, r_halo, hitsPerLayer,
                                 truth, /*scan_gain=*/true,
                                 /*lo=*/0.90, /*hi=*/1.10, /*step=*/0.01,
                                 sigma, /*seed=*/42);
  AxisOptima ax_m = t6_axis_scan(ds, geom, r_core, r_halo, hitsPerLayer,
                                 truth, /*scan_gain=*/false,
                                 /*lo=*/0.75, /*hi=*/0.95, /*step=*/0.01,
                                 sigma, /*seed=*/42);
  double t6_dev = std::max(std::fabs(ax_g.best_mean - truth.gain),
                           std::fabs(ax_m.best_mean - truth.mu));
  char t6note[96];
  std::snprintf(t6note, sizeof(t6note),
                "mean-score optima g=%.2f μ=%.2f; per-event g=%.3f μ=%.3f",
                ax_g.best_mean, ax_m.best_mean,
                ax_g.best_evt, ax_m.best_evt);
  TestResult t6{"T6 Mean-level score is unbiased on the truth axes",
                t6_dev <= 0.015, t6_dev, 0.015, t6note};
  print_result(t6); fails += !t6.ok;
  mark("T6 objective scans (both scores, shared events)");

  write_summary_csv("summary.csv", truth, nominal, ds.target,
                    classical.best_g, classical.best_logL,
                    final_joint, final_joint_L);

  std::printf("\nCSVs written: mu_scan.csv, gscan_nominal_mu.csv, "
              "opt_path.csv, summary.csv\n");
  std::printf("Plot:  python3 concept/plot_oracle_demo.py\n");

  // ----- oracle-call ledger -----------------------------------------------
  std::printf("\n---- Oracle-call ledger ----------------------------------------\n");
  uint64_t total = 0, falsifier = 0;
  for (const auto& e : ledger) total += e.second;
  for (const auto& e : ledger) {
    std::printf("  %-46s %10llu\n", e.first,
                (unsigned long long)e.second);
    // Falsifier side = every phase except building the dataset and the fit
    // itself; the initial J extraction serves the T1/T2 checks.
    std::string label(e.first);
    if (label.rfind("dataset", 0) != 0 && label.rfind("joint fit", 0) != 0)
      falsifier += e.second;
  }
  std::printf("  %-46s %10llu\n", "total",
              (unsigned long long)total);
  std::printf("  falsifier share (all checks / total): %.1f%%\n",
              100.0 * double(falsifier) / double(total));

  std::printf("\n==== %d failure(s) ================================================\n\n",
              fails);
  return fails;
}

} // namespace minlhc

int main() { return minlhc::run(); }
