# Reproducibility Log

Every quantitative claim in `abstract-mode6.md` and in the MODE 6 talk
traces back to a command on this page. The whole chain -- build, run,
plot -- takes under a minute on a laptop. This log records
the exact environment and the verbatim output of the run the numbers were taken
from, so that anyone (including us, later) can tell whether a re-run actually
reproduces them.

> Revised 2026-08-31: T6 (mean-level score control) and the oracle-call
> ledger were added to the demo. T1--T5 values and all four CSVs are
> bit-identical to the originally recorded run; the output below is from
> the revised binary.
>
> Revised 2026-09-02: the plot script gained two slide-sized figures --
> `plot_oracle_window.png` (curves + relative deviation on a shared μ axis,
> 5% tolerance drawn, measured window shaded) and `plot_oracle_payoff.png`
> (the joint trajectory alone, legible at half-slide width); the four
> original figures are drawn by unchanged code. Note on figure bytes: installing extra
> packages into the venv (Pillow, via python-pptx) shifted the PNG encoder,
> so figure files differ at the byte level from the Aug-14 record while the
> plotted content is unchanged; per-run PNG determinism re-verified
> 2026-09-02 (two runs, all six figures bit-identical). The CSVs remain the
> environment-independent invariant.
>
> Revised 2026-09-02, later the same day: the payoff and ensemble figures
> gain deterministic SVG twins for the README (pinned `svg.hashsalt`, no
> `Date` metadata); the checksum list below is refreshed accordingly and
> now also covers the Step-3 sources.

## Environment (recorded 2026-08-14)

| component | version |
| :--- | :--- |
| OS | macOS (Darwin 25.5.0), Apple Silicon |
| C++ compiler | Apple clang 17.0.0 (clang-1700.0.13.5) |
| Python plotting | matplotlib 3.10.8 (in `.demo_venv`) |

Source checksums (SHA-256) of the files that produce the numbers:

```
3db224cfea88cb15f1fc94f58fc644f0306741b3e581820fb337cc90a066378c  concept/opaque_oracle_demo.cpp
20bedd56189cb0593d1f776f5f9793e6a710c7380f3b2b864eb7a9abc9920369  concept/plot_oracle_demo.py
2e9429aba7722cd769f7cd53bfde670e9a4f7c88471ba91ac4cbdf51de4340ba  concept/toy_ensemble.cpp
f24421cf1fe500b0bc112761144a104a73401e42258e6103d09c8af279aba211  concept/bias_check.cpp
434ce96c063d1c3387f50004259a360334e9978a4f5cb0b2dfe27f39f546c42a  concept/plot_toy_ensemble.py
41139105ac7bcb76f6a0c4f7b38f01c1f17453e6795ad979b6f2cdd9f8ae2dc6  concept/plot_score_fix.py
```

## Step 1 -- build and run the demo (~14 s)

```sh
clang++ -std=c++17 -O3 concept/opaque_oracle_demo.cpp -o opaque_oracle_demo
./opaque_oracle_demo
```

Verbatim output of the recorded run (the exit code is the number of failed
diagnostics, so `0` means everything passed):

```
==== Opaque-oracle demo =========================================
Truth:   g=1.000  μ=0.850
Nominal: g=0.850  μ=1.000
Dataset: N=2000 events, 3 hits/layer, core 3x3 + halo 5x5, σ=10.0
Targets at truth: E_core=102.4266  E_halo=30.0554
Extracted J at μ₀=1.000 using 1000 matched-seed oracle runs (ε=2e-02, 239 non-zero cells)

---- Diagnostics -----------------------------------------------
  [OK ] T1a FD vs adjoint dL/dg (fully analytic path)         value=9.919e-13     tol=0.0001
  [OK ] T1b FD vs adjoint dL/dμ (through frozen J)           value=0.2016        tol=0.25      statistical; shrinks as nSeeds↑
  [OK ] T2 Two independent J's agree on L_adj(μ₀+Δμ)     value=0.0036        tol=0.01      two independent J's, nSeeds=1000 each, Δμ=0.020
  [OK ] T3 Linearity window (where adjoint L(μ) ≈ brute)   value=0.08          tol=0.05      μ∈[0.960, 1.080] passes 5% agreement

Classical 1D g-scan (μ held at 1.000): best_g=1.0100  L=-8275.290
Joint optimizer (Adam, 800 iters, J refreshed every 50):
  final g=0.9414  μ=0.8165  L=-4491.941
  diagnostic:  L_brute at truth = -5034.383 (optimum should be ≥ this)
  [OK ] T4 Joint optimizer recovers injected truth            value=0.05863       tol=0.08      stochastic floor ~1/√N
  [OK ] T5 Classical 1D scan is biased (|Δg|/g > 1%)         value=0.01          tol=0.01      lower bound (bias expected)
  [OK ] T6 Mean-level score is unbiased on the truth axes     value=1.11e-16      tol=0.015     mean-score optima g=1.00 μ=0.85; per-event g=0.960 μ=0.810

CSVs written: mu_scan.csv, gscan_nominal_mu.csv, opt_path.csv, summary.csv
Plot:  python3 concept/plot_oracle_demo.py

---- Oracle-call ledger ----------------------------------------
  dataset generation + cache                           4000
  initial J extraction (T1/T2 input)                   2000
  T1a FD + adjoint pass                                6000
  T1b FD + adjoint pass                                6000
  T2 second extraction (predictions: 0 calls)          2000
  T3 brute-force μ-scan (linearized side: 0)         82000
  T5 classical g-scan                                142000
  joint fit (17 J extractions + 800 forwards)       1634000
  truth diagnostic                                     2000
  T6 objective scans (both scores, shared events)      84000
  total                                             1964000
  falsifier share (all checks / total): 16.6%

==== 0 failure(s) ================================================
```

## Step 2 -- regenerate the figures

```sh
python3 concept/plot_oracle_demo.py   # needs matplotlib
```

| CSV | figure | claim it backs (slide / abstract) |
| :--- | :--- | :--- |
| `mu_scan.csv` | `plot_oracle_mu_scan.png` | adjoint L(μ) vs brute-force L(μ) |
| `mu_scan.csv` | `plot_oracle_linearity.png` | measured validity window μ∈[0.96, 1.08] (T3) |
| `mu_scan.csv` | `plot_oracle_window.png` | combined view: both evaluations + deviation, 5% tolerance, shaded window (the talk's RQ3 figure) |
| `gscan_nominal_mu.csv`, `opt_path.csv` | `plot_oracle_gscan_vs_joint.png` | joint fit vs fix-and-scan (T4/T5) |
| `opt_path.csv`, `summary.csv` | `plot_oracle_payoff.png` (+ deterministic `.svg` twin, embedded in the README) | the joint (g, μ) trajectory alone (the talk's payoff figure) |
| `summary.csv` | `plot_oracle_summary.png` | headline truth / classical / joint comparison |

## Step 3 (optional) -- the RQ4 ensemble and score-bias scans

```sh
clang++ -std=c++17 -O3 concept/toy_ensemble.cpp -o toy_ensemble
./toy_ensemble                        # 50 toys; minutes, not seconds
python3 concept/plot_toy_ensemble.py  # -> plot_toy_ensemble.{png,svg}
clang++ -std=c++17 -O3 concept/bias_check.cpp -o bias_check
./bias_check                          # ~7 s; also writes score_scans.csv
python3 concept/plot_score_fix.py     # -> plot_score_fix.png
```

`toy_ensemble` writes `toy_ensemble.csv` and `seeds_scaling.csv`; toy 0
reproduces the exact realization quoted in the talk, and the per-fit
errors come from the Fisher composition `I = J^T I_A J`. `bias_check`
scans the objectives with the optimizer out of the loop and locates the
per-event score's optimum away from the injected truth -- the RQ4
finding; the demo's T6 pins the fixed, mean-level score at truth.
`score_scans.csv` holds both objectives along the truth axes, and
`plot_score_fix.py` draws them side by side: the per-event optimum off
truth at (0.955, 0.810), the mean-level optimum exactly on it.

## Determinism

All random seeds are fixed in the source (`opaque_oracle_demo.cpp`); the demo
is a deterministic function of its code. Checked on 2026-08-14: two independent
runs produced **bit-identical** CSVs (`diff` on all four files) and
byte-identical PNGs. Re-checked 2026-08-31 after the T6/ledger revision:
all four CSVs remain bit-identical to the originally recorded run.

The SVG twins of the two README figures (`plot_oracle_payoff.svg`,
`plot_toy_ensemble.svg`) are likewise bit-identical across runs: the
scripts pin matplotlib's `svg.hashsalt` (element ids otherwise come from
`os.urandom`) and drop the `Date` metadata.

Caveat for other machines: `std::mt19937` is portable, but the algorithm
behind `std::normal_distribution` is implementation-defined
([rand.dist.norm.normal] specifies the distribution, not the method), so exact
digits may differ across C++ standard libraries. The pass/fail verdicts of
T1--T6 are tolerance-based and are expected to hold everywhere; if any fail on
your platform, that is worth reporting, not ignoring.
