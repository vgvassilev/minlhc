// Bias-mechanism check: with the optimizer completely out of the loop,
// where does the demo's mean-target pseudo-likelihood put its optimum?
// Scan g at TRUE mu=0.85, and mu at TRUE g=1.00. If the optima sit away
// from truth, the offset seen in the toy ensemble is a property of the
// objective (per-event (r - T)^2 penalizes model variance), not of the
// optimizer or of any single data realization.
#define main disabled_demo_main
#include "opaque_oracle_demo.cpp"
#undef main

using namespace minlhc;

int main() {
  DetectorGeometry geom{5, 10, 10, 1.0};
  const int hitsPerLayer = 3, r_core = 1, r_halo = 2;
  const double sigma = 10.0;
  Calibration truth; truth.gain = 1.00; truth.mu = 0.85;
  Dataset ds = make_dataset(2000, geom, truth, r_core, r_halo, hitsPerLayer,
                            2024u);

  // g scan at true mu
  double bg = 0, bL = -1e300;
  for (double gg = 0.85; gg <= 1.10 + 1e-12; gg += 0.005) {
    Calibration c; c.gain = gg; c.mu = truth.mu; c.noiseSigma = 0.0;
    double L = forward_logL(ds.particles, geom, c, r_core, r_halo,
                            hitsPerLayer, ds.target, sigma, 42u);
    if (L > bL) { bL = L; bg = gg; }
  }
  std::printf("g scan at true mu=0.85:  best g = %.3f  (truth 1.000)  "
              "L=%.1f\n", bg, bL);

  // mu scan at true g
  double bm = 0; bL = -1e300;
  for (double mm = 0.70; mm <= 1.00 + 1e-12; mm += 0.005) {
    Calibration c; c.gain = truth.gain; c.mu = mm; c.noiseSigma = 0.0;
    double L = forward_logL(ds.particles, geom, c, r_core, r_halo,
                            hitsPerLayer, ds.target, sigma, 42u);
    if (L > bL) { bL = L; bm = mm; }
  }
  std::printf("mu scan at true g=1.00:  best mu = %.3f (truth 0.850)  "
              "L=%.1f\n", bm, bL);

  // joint coarse grid: the pseudo-likelihood's own 2D optimum
  double jg = 0, jm = 0; bL = -1e300;
  for (double gg = 0.88; gg <= 1.02 + 1e-12; gg += 0.005)
    for (double mm = 0.74; mm <= 0.92 + 1e-12; mm += 0.005) {
      Calibration c; c.gain = gg; c.mu = mm; c.noiseSigma = 0.0;
      double L = forward_logL(ds.particles, geom, c, r_core, r_halo,
                              hitsPerLayer, ds.target, sigma, 42u);
      if (L > bL) { bL = L; jg = gg; jm = mm; }
    }
  std::printf("joint grid optimum:      g = %.3f  mu = %.3f  L=%.1f  "
              "(optimizer found g=0.941 mu=0.817)\n", jg, jm, bL);

  // Dump both objectives along the truth axes so the T6 fix can be
  // *shown*, not just printed: per-event score (variance penalty
  // included) vs mean-level score (penalty absent by construction).
  // Same shared-event discipline as the demo's t6_axis_scan.
  FILE* f = std::fopen("score_scans.csv", "w");
  std::fprintf(f, "axis,x,L_event,L_mean\n");
  auto dual = [&](bool scan_gain, double lo, double hi, double step) {
    for (double v = lo; v <= hi + 1e-12; v += step) {
      Calibration c = truth; c.noiseSigma = 0.0;
      (scan_gain ? c.gain : c.mu) = v;
      std::mt19937 rng(42u);
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
      std::fprintf(f, "%s,%.3f,%.6f,%.6f\n", scan_gain ? "g" : "mu", v,
                   evt, -(mc * mc + mh * mh));
    }
  };
  dual(true, 0.85, 1.10, 0.005);
  dual(false, 0.70, 1.00, 0.005);
  std::fclose(f);
  std::printf("CSV written: score_scans.csv (both objectives, truth "
              "axes)\n");
  return 0;
}
