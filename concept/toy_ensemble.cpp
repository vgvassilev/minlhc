// Toy-ensemble study for the MODE 6 talk, reusing the tracked demo verbatim.
//
// Part A: N pseudo-experiments (fresh data realization + fresh model-MC
//         stream per toy; toy 0 = the talk's own realization). For each toy,
//         run the same joint optimizer as the talk, then compute per-fit
//         Fisher uncertainties via the composition I = J^T I_A J (2x2 here)
//         and record fit, errors, and pulls.
// Part B: seeds-scaling of the T1b adjoint-vs-FD relative error
//         (nSeeds in {100, 300, 1000, 3000} x 5 independent extractions)
//         to measure the asserted 1/sqrt(N) shrinkage.
//
// Build:  clang++ -std=c++17 -O3 -I <repo> toy_ensemble.cpp -o toy_ensemble
// Run:    ./toy_ensemble [nToys]        (writes toy_ensemble.csv,
//                                        seeds_scaling.csv to cwd)
#define main disabled_demo_main
#include "opaque_oracle_demo.cpp"
#undef main

#include <cstdlib>

using namespace minlhc;

int main(int argc, char** argv) {
  const int nToys = (argc > 1) ? std::atoi(argv[1]) : 50;

  DetectorGeometry geom{5, 10, 10, 1.0};
  const int hitsPerLayer = 3, r_core = 1, r_halo = 2;
  const double sigma = 10.0;
  Calibration truth;   truth.gain   = 1.00; truth.mu   = 0.85;
  Calibration nominal; nominal.gain = 0.85; nominal.mu = 1.00;
  Particle probe{0.0, 0.0, 0.0, 45.0};
  const double j_eps = 0.02;
  const int j_nSeeds = 1000, N = 2000, nIters = 800, j_refresh = 50;

  // ---- Part A: pseudo-experiment ensemble --------------------------------
  std::ofstream f("toy_ensemble.csv");
  f << std::setprecision(10)
    << "toy,g_fit,mu_fit,g_err,mu_err,rho,pull_g,pull_mu,logL\n";

  for (int t = 0; t < nToys; ++t) {
    // toy 0 reproduces the talk's realization (dataset 2024, model MC 42)
    uint64_t dseed = (t == 0) ? 2024u : 3000u + 17u * uint64_t(t);
    uint64_t mseed = (t == 0) ? 42u   : 7000u + 13u * uint64_t(t);

    Dataset ds = make_dataset(N, geom, truth, r_core, r_halo, hitsPerLayer,
                              dseed);
    OptHistory h = run_joint_optimizer(ds, geom, r_core, r_halo,
                                       hitsPerLayer, nominal, sigma, nIters,
                                       probe, j_eps, j_nSeeds, j_refresh,
                                       mseed);
    Calibration fit;
    fit.gain = h.g.back();
    fit.mu   = h.mu.back();

    // Fisher at the fit point: I = sum_e (v_c v_c^T + v_h v_h^T) / sigma^2
    // with v = (dE_obs/dg, dE_obs/dmu); the mu-derivative comes from a J
    // extracted at the fit point (frozen-operator composition, 2x2 case of
    // I(theta) = J^T I_A J).
    ResponseJacobian Jf = extract_jacobian(probe, geom, hitsPerLayer,
                                           fit.mu, j_eps, j_nSeeds);
    double Jc = 0.0, Jh = 0.0;
    for (const auto& kv : Jf.dE_dmu) {
      int reg = cell_region(kv.first, geom, r_core, r_halo);
      if (reg == 0) Jc += kv.second;
      else if (reg == 1) Jh += kv.second;
    }
    double I00 = 0, I01 = 0, I11 = 0;   // (g,g), (g,mu), (mu,mu)
    std::mt19937 rng(mseed);
    for (const auto& p : ds.particles) {
      auto hits = simulate_shower(p, fit.mu, rng, geom, hitsPerLayer);
      auto digs = digitize(hits, fit, geom, rng);
      double Ec = 0, Eh = 0;   // true energies by region
      for (const auto& d : digs) {
        int reg = cell_region(d.cellId, geom, r_core, r_halo);
        if (reg == 0) Ec += d.trueEnergy;
        else if (reg == 1) Eh += d.trueEnergy;
      }
      double vcg = Ec,            vcm = fit.gain * Jc;
      double vhg = Eh,            vhm = fit.gain * Jh;
      I00 += vcg * vcg + vhg * vhg;
      I01 += vcg * vcm + vhg * vhm;
      I11 += vcm * vcm + vhm * vhm;
    }
    I00 /= sigma * sigma; I01 /= sigma * sigma; I11 /= sigma * sigma;
    double det = I00 * I11 - I01 * I01;
    double eg  = std::sqrt(I11 / det);
    double em  = std::sqrt(I00 / det);
    double rho = -I01 / std::sqrt(I00 * I11);

    double pg = (fit.gain - truth.gain) / eg;
    double pm = (fit.mu   - truth.mu)   / em;
    f << t << "," << fit.gain << "," << fit.mu << "," << eg << "," << em
      << "," << rho << "," << pg << "," << pm << "," << h.logL.back()
      << "\n";
    f.flush();
    std::printf("toy %2d  g=%.4f+-%.4f  mu=%.4f+-%.4f  pull_g=%+.2f  "
                "pull_mu=%+.2f\n", t, fit.gain, eg, fit.mu, em, pg, pm);
  }
  f.close();

  // ---- Part B: seeds-scaling of the T1b relative error -------------------
  Dataset ds0 = make_dataset(N, geom, truth, r_core, r_halo, hitsPerLayer,
                             2024u);
  std::ofstream g("seeds_scaling.csv");
  g << std::setprecision(10) << "nSeeds,rep,t1b_rel_err\n";
  for (int ns : {100, 300, 1000, 3000}) {
    for (int r = 0; r < 5; ++r) {
      ResponseJacobian J = extract_jacobian(probe, geom, hitsPerLayer,
                                            nominal.mu, j_eps, ns,
                                            0xC0FFEEu + uint64_t(r) * 0x10000u);
      TestResult t1 = t1_fd_vs_adjoint_mu(ds0, geom, r_core, r_halo,
                                          hitsPerLayer, nominal, J, sigma,
                                          42u);
      g << ns << "," << r << "," << t1.value << "\n";
      std::printf("seeds %4d rep %d  t1b=%.4f\n", ns, r, t1.value);
    }
  }
  g.close();
  std::printf("wrote toy_ensemble.csv, seeds_scaling.csv\n");
  return 0;
}
