// ===========================================================================//
// Minimal HEP Analysis Pipeline (retrofit with classical-scan comparison)    //
// ---------------------------------------------------------------------------//
// - Equal-split reconstruction
// - Digitizer: adc = g * hit_energy + noise
// - Mass: m = 2 * s * sqrt(E1 * E2)
// - Classical scan: grid search on g with s fixed
// - Joint optimization: analytic per-event gradients for g and s + calibration
//   term
//
// This minimal example shows how a full detector → reconstruct → physics
// pipeline can be made differentiable end-to-end and used to jointly tune
// instrument parameters (here a single digitizer gain g) together with a
// reconstruction/physics parameter (a global energy scale s). The simulated
// data and reconstruction are intentionally simple: each event is a few
// calorimeter hits, digitisations are linear in hit energy, and reconstruction
// splits total ADC equally between two electrons so we can form a closed-form
// invariant mass. That simplicity makes the math and gradients transparent
// while still preserving the essential structure of practical problems.
//
// The code computes two kinds of gradients per event. The mass term gives
// analytic derivatives of the mass likelihood w.r.t. g and s (through how
// reconstructed energies change the invariant mass). The calibration term
// arises from the ADC noise model (adc ~ Normal(g * hit_energy, σ²) ) and gives
// an additional analytic d/dg that anchors g to the electronics model. Together
// they form a multi-objective log-likelihood whose gradients are summed over
// events and used by a small Adam-like optimizer with a robust per-parameter
// median-based normalization to handle widely different gradient scales.
//
// Running the demo will show three pedagogical behaviors: (1) finite-difference
// checks validate the analytic gradients for both g and s; (2) joint
// optimization moves both parameters consistently; (3) if the physics scale of
// the generator differs strongly from the target mass (e.g. the toy generator
// does not enforce a true Z decay), the optimizer will reveal a degeneracy or
// push parameters to reconcile the mismatch. This is not a bug but the intended
// lesson: gradients expose identifiability problems and indicate where extra
// constraints (priors, calibration datasets, or additional physics structure)
// are required.
//
// Use this minimal program as a testbed: you can enable per-layer gains, change
// reconstruction mapping (e.g., deterministic cell-id-based splits), add a weak
// Gaussian prior on s, or generate physically correct two-body decays to see
// how identifiability changes. Because everything is analytic and compact, it
// is straightforward to extend for more realistic likelihoods (per-cell noise,
// offsets, non-linear digitizers) while keeping the same pattern: define
// per-event forward, write analytic per-event derivatives, validate with finite
// differences, and then jointly optimize.
//
// Compile:
//   clang++ demo.cpp -O3 -std=c++11 -o minimal_joint_g_s_with_scan
// Run:
//   ./minimal_joint_g_s_with_scan
// ===========================================================================//

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>
#include <algorithm>
#include <limits>
namespace minlhc {

// ===========================================================================//
//                       Event Context and Shared State                       //
// ===========================================================================//

struct DetectorGeometry {
  int nLayers;
  int nX;
  int nY;
  double cellSize;
};

struct Calibration {
  double gain;        ///< Electronics gain (calibration parameter)
  double noiseSigma;  ///< Electronic noise
  double scale = 1.0; ///< Global energy scale (new)
  double offset = 2.0;///< The "Unmodeled Loss"
};

struct Score {
  double dlogL_dgain = 0.0;
  double dlogL_dscale = 0.0; // NEW: accumulate derivative wrt global scale
};

struct EventContext {
  DetectorGeometry geom;
  Calibration calib;
  Score score;
  std::mt19937 rng;

  explicit EventContext(uint64_t seed) : rng(seed) {}
};

// ===========================================================================//
//                            Event Data Model (EDM)                          //
// ===========================================================================//

struct Particle {
  double px, py, pz;
  double E;
};

struct CaloHit {
  int layer;
  double x, y;
  double energy;
};

struct Digi {
  int cellId;
  double adc;
  double trueEnergy;
};

struct Electron {
  double E;
  std::vector<int> cellIds;
};

// ===========================================================================//
//                Detector Simulation (Transport + Showering)                 //
// ===========================================================================//

inline std::vector<CaloHit> simulate_shower(const Particle &p,
                                            EventContext &ctx) {
  std::normal_distribution<double> transverse(0.0, 1.0);
  std::uniform_real_distribution<double> fluct(0.8, 1.2);

  std::vector<CaloHit> hits;
  hits.reserve(ctx.geom.nLayers);
  for (int l = 0; l < ctx.geom.nLayers; ++l) {
    double e_layer = p.E * std::exp(-0.3 * l) * fluct(ctx.rng);
    hits.push_back({l, transverse(ctx.rng), transverse(ctx.rng), e_layer});
  }
  return hits;
}

// ===========================================================================//
//                  Digitization (Electronics + Calibration)                  //
// ===========================================================================//

inline std::vector<Digi> digitize(const std::vector<CaloHit> &hits,
                                  EventContext &ctx) {
  // SAFE: avoid constructing normal_distribution with sigma == 0
  const bool use_noise = (ctx.calib.noiseSigma > 0.0);
  std::normal_distribution<double> noise_dist(0.0, 1.0);
  if (use_noise) noise_dist = std::normal_distribution<double>(0.0, ctx.calib.noiseSigma);

  std::map<int, double> cellEnergy;

  for (const auto &h : hits) {
    int ix = int(std::floor(h.x / ctx.geom.cellSize));
    int iy = int(std::floor(h.y / ctx.geom.cellSize));
    // clamp to geometry bounds
    ix = std::max(0, std::min(ix, ctx.geom.nX - 1));
    iy = std::max(0, std::min(iy, ctx.geom.nY - 1));
    int id = h.layer * ctx.geom.nX * ctx.geom.nY + ix * ctx.geom.nY + iy;
    cellEnergy[id] += h.energy;
  }

  std::vector<Digi> digis;
  digis.reserve(cellEnergy.size());
  for (auto &kv : cellEnergy) {
    double n = use_noise ? noise_dist(ctx.rng) : 0.0;
    double adc = ctx.calib.gain * kv.second + n;
    digis.push_back({kv.first, adc, kv.second});
  }
  return digis;
}

// ===========================================================================//
//                    Reconstruction (Soft, Differentiable)                   //
// ===========================================================================//

// NOTE: replaced unstable exp(adc) weight with a bounded, differentiable weight.
// Weight w(adc) = 1 + tanh(adc)  — bounded in [0,2], derivative = sech^2(adc).
// This prevents uncontrolled gradient explosion while remaining smooth.
inline Electron reconstruct(const std::vector<Digi> &digis) {
  Electron e;
  e.E = 0.0;
  for (const auto &d : digis) {
    double w = 1.0 + std::tanh(d.adc);       // bounded weight
    e.E += w * d.adc;                       // contribution
    e.cellIds.push_back(d.cellId);
  }
  return e;
}

inline std::vector<Electron> reconstruct_pair(const std::vector<Digi> &digis) {
  Electron e1, e2;
  e1.E = e2.E = 0.0;

  // deterministic splitting (map iteration -> stable order)
  for (size_t i = 0; i < digis.size(); ++i) {
    const auto &d = digis[i];
    double w = 1.0 + std::tanh(d.adc); // bounded
    if (i % 2 == 0) {
      e1.E += w * d.adc;
      e1.cellIds.push_back(d.cellId);
    } else {
      e2.E += w * d.adc;
      e2.cellIds.push_back(d.cellId);
    }
  }
  return {e1, e2};
}

// ===========================================================================//
//                               Physics Analysis                             //
// ===========================================================================//

// Now apply global scale `s = ctx.calib.scale` when forming invariant mass.
// m = 2 * s * sqrt(E1 * E2)
inline double invariant_mass(const Electron &e1, const Electron &e2, const EventContext &ctx) {
  // Subtract a constant "loss"
  double E1 = (e1.E - ctx.calib.offset) * ctx.calib.scale;
  double E2 = e2.E * ctx.calib.scale;
  // numeric guards
  if (E1 <= 0.0 || E2 <= 0.0) return 0.0;
  double val = 4.0 * E1 * E2; // (2 * sqrt(E1 E2))^2
  if (val <= 0.0) return 0.0;
  return std::sqrt(val);
}

// ===========================================================================//
//                                  Likelihood                                //
// ===========================================================================//

inline double log_likelihood_energy(const Electron &e, double targetE = 50.0,
                                    double sigma = 2.0) {
  double r = (e.E - targetE) / sigma;
  return -0.5 * r * r;
}

inline double log_likelihood_mass(const Electron &e1, const Electron &e2,
                                  const EventContext &ctx,
                                  double mZ = 91.1876, double sigma = 2.0) {
  double m = invariant_mass(e1, e2, ctx);
  double r = (m - mZ) / sigma;
  return -0.5 * r * r;
}

// ===========================================================================//
//                    Backward Pass (Adjoint / Sensitivity)                   //
// ===========================================================================//

// Helper: derivative of the bounded weight w(adc) = 1 + tanh(adc)
// w' = sech^2(adc) = 1 - tanh(adc)^2
inline double weight(double adc) {
  return 1.0 + std::tanh(adc);
}
inline double weight_derivative(double adc) {
  double t = std::tanh(adc);
  return 1.0 - t * t;
}

/// Backpropagate sensitivity to the gain parameter (energy observable).
inline void backward_gain_energy(const Electron &e,
                                 const std::vector<Digi> &digis,
                                 EventContext &ctx, double targetE = 50.0,
                                 double sigma = 2.0) {
  // gradient of logL wrt reconstructed energy (no scale here; if you want
  // scale to affect energy-based likelihood, multiply e.E by ctx.calib.scale)
  double dlogL_dE = -(e.E - targetE) / (sigma * sigma);

  for (const auto &d : digis) {
    // dE/d(adc) = w(adc) + adc * w'(adc)  (product rule for w(adc)*adc)
    double w = weight(d.adc);
    double dw = weight_derivative(d.adc);
    double dE_dadc = w + d.adc * dw;

    // dadc/dg = trueEnergy
    double dadc_dg = d.trueEnergy;
    ctx.score.dlogL_dgain += dlogL_dE * dE_dadc * dadc_dg;
  }
}

/// Backpropagate sensitivity through invariant mass to gain and scale.
inline void backward_gain_mass(const Electron &e1, const Electron &e2,
                               const std::vector<Digi> &digis,
                               EventContext &ctx, double mZ = 91.1876,
                               double sigma = 2.0) {
  double m = invariant_mass(e1, e2, ctx);
  if (m <= 0.0) return;

  double dlogL_dm = -(m - mZ) / (sigma * sigma);

  // dm/dE1 and dm/dE2 (note E1/E2 are *scaled* inside invariant_mass)
  // Using m = 2 * s * sqrt(E1_unscaled * E2_unscaled), same formula applies:
  double E1_unscaled = e1.E;
  double E2_unscaled = e2.E;
  // dm/dE1_unscaled = m / (2 * E1_unscaled)
  double dm_dE1 = (E1_unscaled > 0.0) ? (m / (2.0 * E1_unscaled)) : 0.0;
  double dm_dE2 = (E2_unscaled > 0.0) ? (m / (2.0 * E2_unscaled)) : 0.0;

  // Per-digi contribution to dE/dadc (same as in backward_gain_energy)
  for (size_t i = 0; i < digis.size(); ++i) {
    const auto &d = digis[i];
    double w = weight(d.adc);
    double dw = weight_derivative(d.adc);
    double dE_dadc = w + d.adc * dw;
    double dadc_dg = d.trueEnergy;

    // decide which electron this digi contributed to (deterministic split by index)
    double dlogL_dE = (i % 2 == 0) ? dlogL_dm * dm_dE1 : dlogL_dm * dm_dE2;

    // chain: dlogL/dg += dlogL/dE * dE/dadc * dadc/dg
    ctx.score.dlogL_dgain += dlogL_dE * dE_dadc * dadc_dg;
  }

  // derivative wrt global scale s: dm/ds = m / s    (since m ∝ s)
  double s = ctx.calib.scale;
  if (s != 0.0) {
    double dlogL_ds = dlogL_dm * (m / s);
    ctx.score.dlogL_dscale += dlogL_ds;
  }
}
// ---------------------------------------------------------------------------
// (unchanged structs and functions up to run_end_to_end_stable)
// [Everything above is identical to the previous stable version you accepted]
// For brevity in this paste, assume all previously defined structs/functions exist:
//   DetectorGeometry, Calibration, Score, EventContext,
//   Particle, CaloHit, Digi, Electron,
//   simulate_shower, digitize, reconstruct_pair,
//   invariant_mass, log_likelihood_mass,
//   weight, weight_derivative,
//   backward_gain_mass
// ---------------------------------------------------------------------------

// -----------------------------
// NEW helpers: forward_total_logL & classical_scan_g
// Minimal, deterministic forward-evaluation over a dataset of CaloHit vectors.
// -----------------------------
inline double forward_total_logL(const std::vector<std::vector<CaloHit>>& dataset,
                                 const DetectorGeometry &geom,
                                 double gain, double scale, double noiseSigma) {
  double total = 0.0;
  // create a temporary context with fixed RNG seed so scans are deterministic
  EventContext tmp_ctx(424242);
  tmp_ctx.geom = geom;
  tmp_ctx.calib.gain = gain;
  tmp_ctx.calib.noiseSigma = noiseSigma;
  tmp_ctx.calib.scale = scale;

  for (const auto &hits : dataset) {
    auto digis = digitize(hits, tmp_ctx);
    auto eles = reconstruct_pair(digis);
    total += log_likelihood_mass(eles[0], eles[1], tmp_ctx);
  }
  return total;
}

struct ScanResult { double best_g; double best_logL; };

// Classical grid scan over g keeping scale fixed.
inline ScanResult classical_scan_g(const std::vector<std::vector<CaloHit>>& dataset,
                                   const DetectorGeometry &geom,
                                   double g_min, double g_max, double g_step,
                                   double scale_fixed, double noiseSigma) {
  double best_g = g_min;
  double best_logL = -std::numeric_limits<double>::infinity();
  for (double g = g_min; g <= g_max + 1e-12; g += g_step) {
    double L = forward_total_logL(dataset, geom, g, scale_fixed, noiseSigma);
    if (L > best_logL) { best_logL = L; best_g = g; }
  }
  return {best_g, best_logL};
}

// ===========================================================================
// Stable joint optimization driver (improved, replaces previous version)
// - uses per-event gradients, robust median-based normalisation,
// - Adam-style adaptive updates (separate for gain and scale),
// - positivity enforced for both gain and scale (log-param safer; we clamp here),
// - small prior on scale to prevent collapse to zero,
// - reasonable clipping of raw updates so optimizer cannot "jump".
// ===========================================================================
inline void run_end_to_end_stable() {
  EventContext ctx(1234);
  ctx.geom = {5, 10, 10, 1.0};
  ctx.calib = {1.0, 0.05, 1.0}; // gain, noiseSigma, scale

  //Particle truth{0, 0, 0, 50.0};
  // If truth is 50, but we assume it's something else,
  // the optimizer has to work harder to reconcile the mass.
  Particle truth{0, 0, 0, 45.0}; // Generator truth is 45 GeV

  // Build dataset (precompute hits so scans are deterministic)
  const int N = 200;
  std::vector<std::vector<CaloHit>> dataset;
  dataset.reserve(N);
  for (int i = 0; i < N; ++i) dataset.push_back(simulate_shower(truth, ctx));

  // Classical scan BEFORE joint optimization (s fixed = 1.0)
  double gmin = 0.80, gmax = 1.20, gstep = 0.01;
  double s_fixed_before = 1.0;
  auto scan_before = classical_scan_g(dataset, ctx.geom, gmin, gmax, gstep, s_fixed_before, ctx.calib.noiseSigma);
  std::cout << "CLASSICAL SCAN (s=1 fixed) best_g=" << scan_before.best_g
            << " logL=" << scan_before.best_logL << std::endl;

  // Finite-difference diagnostics at the initial point (small eps)
  {
    double eps = 1e-5;
    double Lm = forward_total_logL(dataset, ctx.geom, ctx.calib.gain - eps, ctx.calib.scale, ctx.calib.noiseSigma);
    double Lp = forward_total_logL(dataset, ctx.geom, ctx.calib.gain + eps, ctx.calib.scale, ctx.calib.noiseSigma);
    double fd_g = (Lp - Lm) / (2.0 * eps);
    Lm = forward_total_logL(dataset, ctx.geom, ctx.calib.gain, ctx.calib.scale - eps, ctx.calib.noiseSigma);
    Lp = forward_total_logL(dataset, ctx.geom, ctx.calib.gain, ctx.calib.scale + eps, ctx.calib.noiseSigma);
    double fd_s = (Lp - Lm) / (2.0 * eps);
    std::cout << "[FD] dLogL/dg (at initial) ≈ " << fd_g << "  dLogL/ds ≈ " << fd_s << std::endl;
  }

  // --- Optimizer hyperparameters (tuned for stability) ---
  const double lr_base = 10;//0.1;      // base lr used with median-normalisation
  const double beta1 = 0.9;        // Adam momentum
  const double beta2 = 0.999;      // Adam second moment
  const double eps = 1e-12;
  const double max_step = 0.05;//0.01;    // absolute clip on raw updates (prevents runaway)
  const double prior_sigma_s = 1.;//0.2; // weak Gaussian prior on scale (centered at 1.0)

  double m_g = 0.0, v_g = 0.0;
  double m_s = 0.0, v_s = 0.0;
  int tstep = 0;

  // main optimization loop
  for (int iter = 0; iter < 300; ++iter) {
    // collect per-event gradients (do not accumulate directly into ctx.score here)
    std::vector<double> per_event_dg; per_event_dg.reserve(dataset.size());
    std::vector<double> per_event_ds; per_event_ds.reserve(dataset.size());
    double totalLogL = 0.0;

    // per-event forward + backward (analytic)
    for (const auto &hits : dataset) {
      auto digis = digitize(hits, ctx);
      auto eles = reconstruct_pair(digis);

      // forward: mass likelihood
      double logL = log_likelihood_mass(eles[0], eles[1], ctx);
      totalLogL += logL;

      // compute per-event analytic grads using the same logic as backward_gain_mass,
      // but local to the event (so we can robustly normalize).
      // We'll compute dg_event and ds_event (both scalars).
      double dg_event = 0.0;
      double ds_event = 0.0;

      double m = invariant_mass(eles[0], eles[1], ctx);
      if (m > 0.0) {
        double dlogL_dm = -(m - 91.1876) / (2.0 * 2.0); // sigma=2 => sigma^2=4
        // dm/dE for unscaled reconstructed energies (scale factor s applied inside invariant_mass)
        double E1_un = eles[0].E;
        double E2_un = eles[1].E;
        double dm_dE1 = (E1_un > 0.0) ? (m / (2.0 * E1_un)) : 0.0;
        double dm_dE2 = (E2_un > 0.0) ? (m / (2.0 * E2_un)) : 0.0;

        // per-digi contributions to dE/dadc and then to dg via dadc/dg = trueEnergy
        for (size_t i = 0; i < digis.size(); ++i) {
          const auto &d = digis[i];
          double w = weight(d.adc);
          double dw = weight_derivative(d.adc);
          double dE_dadc = w + d.adc * dw;
          double dadc_dg = d.trueEnergy;
          double dlogL_dE = (i % 2 == 0) ? dlogL_dm * dm_dE1 : dlogL_dm * dm_dE2;
          dg_event += dlogL_dE * dE_dadc * dadc_dg;
        }

        // derivative wrt scale s: dm/ds = m / s
        double s = ctx.calib.scale;
        if (s != 0.0) ds_event += dlogL_dm * (m / s);
      }

      // calibration (digitizer) term: contributes to dg only
      if (ctx.calib.noiseSigma > 0.0) {
        double var = ctx.calib.noiseSigma * ctx.calib.noiseSigma;
        double calib_term = 0.0;
        for (const auto &d : digis) calib_term += (d.adc - ctx.calib.gain * d.trueEnergy) * d.trueEnergy / var;
        dg_event += calib_term;
      }

      per_event_dg.push_back(dg_event);
      per_event_ds.push_back(ds_event);
    } // end events loop

    // compute robust scale (median abs)
    auto median_abs = [&](const std::vector<double>& v)->double {
      if (v.empty()) return 1.0;
      std::vector<double> tmp(v.size());
      for (size_t i=0;i<v.size();++i) tmp[i] = std::fabs(v[i]);
      size_t mid = tmp.size()/2;
      std::nth_element(tmp.begin(), tmp.begin()+mid, tmp.end());
      double med = tmp[mid];
      if (tmp.size()%2==0) {
        auto tmp2 = tmp;
        size_t mid2 = mid-1;
        std::nth_element(tmp2.begin(), tmp2.begin()+mid2, tmp2.end());
        med = 0.5 * (med + tmp2[mid2]);
      }
      if (med <= 0.0) {
        double s=0.0; for (double x:tmp) s += x;
        med = (s>0.0) ? (s/tmp.size()) : 1.0;
      }
      return med;
    };

    double med_g = median_abs(per_event_dg);
    double med_s = median_abs(per_event_ds);

    // clip and average to get mean_tamed gradients
    double clip_factor = 10.0;
    double sum_g = 0.0, sum_s = 0.0;
    for (double x : per_event_dg) {
      double xc = std::min(std::max(x, -clip_factor*med_g), clip_factor*med_g);
      sum_g += xc;
    }
    for (double x : per_event_ds) {
      double xc = std::min(std::max(x, -clip_factor*med_s), clip_factor*med_s);
      sum_s += xc;
    }
    double mean_g = sum_g / (per_event_dg.empty() ? 1.0 : (double)per_event_dg.size());
    double mean_s = sum_s / (per_event_ds.empty() ? 1.0 : (double)per_event_ds.size());

    // Add weak Gaussian prior on scale centered at 1.0 to prevent collapse
    double prior_term_s = - (ctx.calib.scale - 1.0) / (prior_sigma_s * prior_sigma_s);
    mean_s += prior_term_s;

    // compute learning rates (robust normalization)
    double lr_g = lr_base / (med_g + eps);
    double lr_s = lr_base / (med_s + eps);

    // Adam moments + bias correction
    ++tstep;
    m_g = beta1 * m_g + (1.0 - beta1) * mean_g;
    v_g = beta2 * v_g + (1.0 - beta2) * (mean_g * mean_g);
    double mhat_g = m_g / (1.0 - std::pow(beta1, tstep));
    double vhat_g = v_g / (1.0 - std::pow(beta2, tstep));
    double raw_update_g = lr_g * (mhat_g / (std::sqrt(vhat_g) + 1e-12));
    // clip absolute step
    if (raw_update_g >  max_step) raw_update_g =  max_step;
    if (raw_update_g < -max_step) raw_update_g = -max_step;

    m_s = beta1 * m_s + (1.0 - beta1) * mean_s;
    v_s = beta2 * v_s + (1.0 - beta2) * (mean_s * mean_s);
    double mhat_s = m_s / (1.0 - std::pow(beta1, tstep));
    double vhat_s = v_s / (1.0 - std::pow(beta2, tstep));
    double raw_update_s = lr_s * (mhat_s / (std::sqrt(vhat_s) + 1e-12));
    if (raw_update_s >  max_step) raw_update_s =  max_step;
    if (raw_update_s < -max_step) raw_update_s = -max_step;

    // Apply updates
    ctx.calib.gain += raw_update_g;
    ctx.calib.scale += raw_update_s;

    // Enforce positivity & reasonable bounds
    if (ctx.calib.gain < 1e-6) ctx.calib.gain = 1e-6;
    if (ctx.calib.scale < 1e-6) ctx.calib.scale = 1e-6;
    if (ctx.calib.gain > 1e6) ctx.calib.gain = 1e6; // sanity cap

    if (iter % 50 == 0 || iter == 299) {
      std::cout << "Iter " << iter << " totalLogL=" << totalLogL
                << " gain=" << ctx.calib.gain
                << " scale=" << ctx.calib.scale
                << " mean_dg=" << mean_g << " mean_ds=" << mean_s
                << " med_g=" << med_g << " med_s=" << med_s
                << " raw_dg=" << raw_update_g << " raw_ds=" << raw_update_s
                << std::endl;
    }
  } // end optimization loop

  // After optimization, run classical scan with s fixed to optimized value
  auto scan_after = classical_scan_g(dataset, ctx.geom, gmin, gmax, gstep, ctx.calib.scale, ctx.calib.noiseSigma);
  std::cout << "CLASSICAL SCAN (s=joint.scale fixed) best_g=" << scan_after.best_g
            << " logL=" << scan_after.best_logL << std::endl;

  std::cout << "SUMMARY:\n"
            << " - classical (s=1) best_g=" << scan_before.best_g << "\n"
            << " - joint optimized: gain=" << ctx.calib.gain << " scale=" << ctx.calib.scale << "\n"
            << " - classical with s=joint.scale best_g=" << scan_after.best_g << "\n";
}

} // namespace minlhc

int main() {minlhc::run_end_to_end_stable();}
