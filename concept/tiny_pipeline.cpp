/// ============================================================================
/// Minimal HEP Analysis Pipeline
/// ============================================================================
///
/// This header provides a minimal, end-to-end, differentiable HEP analysis
/// pipeline, written entirely in C++ and designed for conceptual clarity.
///
/// The pipeline mirrors the structure of real LHC experiments:
///
///   - theory / generator (truth-level particles)
///   - detector simulation (transport + energy deposition)
///   - digitization (electronics + calibration)
///   - reconstruction (physics objects)
///   - analysis (likelihood-based inference)
///   - [NEW] backward / adjoint pass (sensitivity extraction, gradients)
///
/// Each component maps directly onto a real subsystem in a production
/// experiment (CMS / ATLAS / LHCb), while remaining small enough to understand.
///
/// The backward pass is written explicitly to illustrate how gradients
/// propagate through the detector and analysis chain -- analogous to
/// differentiable ray tracing in computer graphics.
///
/// This code is NOT a replacement for Geant4. Instead, it demonstrates how
/// low-dimensional detector sensitivity operators can be extracted and
/// composed with inference.
///
///\license This project is released under the Apache 2.0 license.
///\copyright Copyright 2025 Vassil Vassilev.
/// ============================================================================

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>

namespace minlhc {

// ===========================================================================//
//                       Event Context and Shared State                       //
// ===========================================================================//

/// Detector geometry description.
///
/// \remark Real equivalent: ECAL / HGCAL geometry (CMS); LAr / Tile geometry
///         (ATLAS)
///
struct DetectorGeometry {
  int nLayers;     ///< Number of calorimeter layers
  int nX;          ///< Cells in x
  int nY;          ///< Cells in y
  double cellSize; ///< Cell granularity
};

/// Detector calibration parameters.
///
/// \remark Typically stored in conditions databases and optimized
///         using calibration workflows.
///
struct Calibration {
  double gain;       ///< Electronics gain (calibration parameter)
  double noiseSigma; ///< Electronic noise
};

/// Gradient (score) accumulator.
///
/// Stores derivatives of the log-likelihood with respect to detector
/// or physics parameters.
///
struct Score {
  double dlogL_dgain = 0.0;
};

/// Central event context (analogous to CMSSW EventSetup).
///
/// Holds geometry, calibration, RNG state, and accumulated gradients.
///
/// \remark Central RNG control is essential: gradients are only meaningful
///         if stochastic paths are held fixed during forward/backward passes.
///
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

/// Generator-level particle.
///
/// \remark Real equivalent: HepMC::GenParticle; reco::GenParticle
///
struct Particle {
  double px, py, pz;
  double E;
};

/// Energy deposition in detector material.
///
/// \remark Real equivalent: Geant4 step aggregation; PCaloHit
///
struct CaloHit {
  int layer;
  double x, y;
  double energy;
};

/// Digitized detector readout.
///
/// \note trueEnergy is retained explicitly to enable gradient propagation;
///       in real experiments this information is usually discarded.
///
/// \remark Real equivalent: EcalDigi; HGCalDigi
///
struct Digi {
  int cellId;
  double adc;
  double trueEnergy;
};

/// Reconstructed electron candidate.
///
/// \remark Real equivalent: reco::GsfElectron; ATLAS electron objects
///
struct Electron {
  double E;
  std::vector<int> cellIds;
};

// ===========================================================================//
//                Detector Simulation (Transport + Showering)                 //
// ===========================================================================//

/// Simulate an electromagnetic shower.
///
/// Parametric, differentiable stand-in for full particle transport.
///
/// \remark Conceptual equivalent: MadGraph / Pythia → Geant4 or HepEm
///
inline std::vector<CaloHit> simulate_shower(const Particle &p,
                                            EventContext &ctx) {
  std::normal_distribution<double> transverse(0.0, 1.0);
  std::uniform_real_distribution<double> fluct(0.8, 1.2);

  std::vector<CaloHit> hits;
  for (int l = 0; l < ctx.geom.nLayers; ++l) {
    double e_layer = p.E * std::exp(-0.3 * l) * fluct(ctx.rng);
    hits.push_back({l, transverse(ctx.rng), transverse(ctx.rng), e_layer});
  }
  return hits;
}

// ===========================================================================//
//                  Digitization (Electronics + Calibration)                  //
// ===========================================================================//

/// Digitize energy deposits into electronic signals.
///
/// \remark Real equivalent: EcalElectronicsSim (CMS); LAr digitization (ATLAS)
///
inline std::vector<Digi> digitize(const std::vector<CaloHit> &hits,
                                  EventContext &ctx) {
  std::normal_distribution<double> noise(0.0, ctx.calib.noiseSigma);
  std::map<int, double> cellEnergy;

  for (const auto &h : hits) {
    int ix = int(h.x / ctx.geom.cellSize);
    int iy = int(h.y / ctx.geom.cellSize);
    int id = h.layer * ctx.geom.nX * ctx.geom.nY + ix * ctx.geom.nY + iy;
    cellEnergy[id] += h.energy;
  }

  std::vector<Digi> digis;
  for (auto &kv : cellEnergy) {
    double adc = ctx.calib.gain * kv.second + noise(ctx.rng);
    digis.push_back({kv.first, adc, kv.second});
  }
  return digis;
}

// ===========================================================================//
//                    Reconstruction (Soft, Differentiable)                   //
// ===========================================================================//

/// Reconstruct a single electron candidate from digis.
///
/// Intentionally trivial (all digis → one electron).
///
/// \remark Real equivalent: clustering; track–cluster matching; electron
///         identification
///
inline Electron reconstruct(const std::vector<Digi> &digis) {
  Electron e;
  e.E = 0.0;

  for (const auto &d : digis) {
    double w = std::exp(d.adc); // soft weight
    e.E += w * d.adc;
    e.cellIds.push_back(d.cellId);
  }
  return e;
}

/// Reconstruct a pair of electrons from digis.
///
/// \remark Pedagogical stand-in for clustering + pairing.
///         Digis are split deterministically to preserve differentiability.
inline std::vector<Electron> reconstruct_pair(const std::vector<Digi> &digis) {
  Electron e1, e2;
  e1.E = e2.E = 0.0;

  for (size_t i = 0; i < digis.size(); ++i) {
    const auto &d = digis[i];
    double w = std::exp(d.adc);

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

/// Compute invariant mass of two reconstructed electrons.
///
/// Minimal kinematic model where electrons are assumed back-to-back along z.
/// Sufficient to demonstrate sensitivity propagation.
///
inline double invariant_mass(const Electron &e1, const Electron &e2) {
  double E = e1.E + e2.E;
  double pz = e1.E - e2.E;
  return std::sqrt(E * E - pz * pz);
}

// ===========================================================================//
//                                  Likelihood                                //
// ===========================================================================//

/// Unbinned Gaussian log-likelihood on electron energy.
///
/// \note Histogramming is intentionally avoided:
///       binning destroys differentiability and discards information.
inline double log_likelihood_energy(const Electron &e, double targetE = 50.0,
                                    double sigma = 2.0) {
  double r = (e.E - targetE) / sigma;
  return -0.5 * r * r;
}

/// Unbinned Gaussian log-likelihood on invariant mass (Z → ee).
inline double log_likelihood_mass(const Electron &e1, const Electron &e2,
                                  double mZ = 91.1876, double sigma = 2.0) {
  double m = invariant_mass(e1, e2);
  double r = (m - mZ) / sigma;
  return -0.5 * r * r;
}

// ===========================================================================//
//                    Backward Pass (Adjoint / Sensitivity)                   //
// ---------------------------------------------------------------------------//
// WARNING:
// The exponential weighting below is intentionally unphysical and is used only
// to demonstrate how unstable reconstruction nonlinearities can lead to
// gradient explosion if not carefully controlled. Real HEP reconstruction
// algorithms employ bounded or normalized weights.
// ===========================================================================//

/// Backpropagate sensitivity to the gain parameter (energy observable).
inline void backward_gain_energy(const Electron &e,
                                 const std::vector<Digi> &digis,
                                 EventContext &ctx, double targetE = 50.0,
                                 double sigma = 2.0) {
  double dlogL_dE = -(e.E - targetE) / (sigma * sigma);

  for (const auto &d : digis) {
    double dE_dadc = std::exp(d.adc);
    double dadc_dg = d.trueEnergy;
    ctx.score.dlogL_dgain += dlogL_dE * dE_dadc * dadc_dg;
  }
}

/// Backpropagate sensitivity through invariant mass to gain.
inline void backward_gain_mass(const Electron &e1, const Electron &e2,
                               const std::vector<Digi> &digis,
                               EventContext &ctx, double mZ = 91.1876,
                               double sigma = 2.0) {
  double m = invariant_mass(e1, e2);
  double dlogL_dm = -(m - mZ) / (sigma * sigma);

  double dm_dE1 = e1.E / m;
  double dm_dE2 = e2.E / m;

  for (size_t i = 0; i < digis.size(); ++i) {
    const auto &d = digis[i];
    double dE_dadc = std::exp(d.adc);
    double dadc_dg = d.trueEnergy;

    double dlogL_dE = (i % 2 == 0) ? dlogL_dm * dm_dE1 : dlogL_dm * dm_dE2;

    ctx.score.dlogL_dgain += dlogL_dE * dE_dadc * dadc_dg;
  }
}

// ===========================================================================//
//                  Example Driver (Calibration Optimization)                 //
// ===========================================================================//

/// Example calibration loop using gradient ascent.
///
/// Demonstrates end-to-end forward + backward optimization.
///
inline void run_end_to_end_unstable() {
  EventContext ctx(1234);
  ctx.geom = {5, 10, 10, 1.0};
  ctx.calib = {1.0, 0.1};

  Particle truth{0, 0, 50, 50};
  double lr = 0.01;

  for (int iter = 0; iter < 10; ++iter) {
    ctx.score.dlogL_dgain = 0.0;

    auto hits = simulate_shower(truth, ctx);
    auto digis = digitize(hits, ctx);
    auto eles = reconstruct_pair(digis);

    double logL = log_likelihood_mass(eles[0], eles[1]);
    //
    // The terms ctx.score.dlogL_dgain += dlogL_dE * dE_dadc * dadc_dg; will
    // often yield:
    // Iter 0 logL=-3.05929e+42 gain=-1.45007e+44;
    // Iter 1 logL=-1039.4 gain=-nan
    // ...
    //
    // ∂logL/∂g ∝ exp(adc) => exp(adc + δ) = exp(adc) · exp(δ) which implies
    // positive feedback not optimization.
    //
    // -------------------------------------------------------------------------
    //
    // Backward (adjoint) pass through the invariant-mass analysis
    //
    // This call propagates sensitivity information from the final physics
    // objective (the likelihood of the Z → ee invariant mass) back through the
    // reconstruction, digitization, and detector calibration layers.
    //
    // Conceptually, this computes the gradient:
    //
    //   ∂ log L(m_ee) / ∂ g
    //
    // where g is the calorimeter gain calibration parameter.
    //
    // Unlike the classical workflow -- where gain parameters are tuned by
    // repeatedly regenerating histograms and manually inspecting peak position
    // and width -- this adjoint pass provides the exact per-event sensitivity
    // of the physics likelihood to detector parameters.
    //
    // Each term in the backward pass corresponds to a physical dependency:
    //   * how invariant mass depends on reconstructed electron energies,
    //   * how reconstructed energies depend on digitized signals,
    //   * how digitized signals depend on the gain calibration.
    //
    // The result is an accumulated score (gradient) stored in ctx.score, which
    // can be used to update detector parameters via gradient ascent/descent.
    //
    // This is directly analogous to differentiable ray tracing: instead of
    // asking "what image do I get?", we ask "how must the scene change to
    // improve the objective?" -- here, the objective is physics sensitivity.
    //
    // Importantly, this operates at the event level, avoids histogram binning,
    // and captures small effects that would be invisible to brute-force scans.
    //
    // This step is the core mechanism that enables end-to-end detector-aware
    // optimization of physics analyses.
    //
    // -------------------------------------------------------------------------
    backward_gain_mass(eles[0], eles[1], digis, ctx);

    ctx.calib.gain += lr * ctx.score.dlogL_dgain;

    std::cout << "Iter " << iter << " logL=" << logL
              << " gain=" << ctx.calib.gain << std::endl;
  }
}

} // namespace minlhc
