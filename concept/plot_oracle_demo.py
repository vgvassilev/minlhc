#!/usr/bin/env python3
# ============================================================================
# Plots for concept/opaque_oracle_demo.cpp
#
# Run AFTER the C++ demo has written mu_scan.csv, gscan_nominal_mu.csv,
# opt_path.csv, and summary.csv to the current working directory.
#
# Produces five PNGs that together make the pedagogical argument:
#
#   plot_oracle_mu_scan.png       — adjoint L(μ) vs brute-force L(μ)
#   plot_oracle_linearity.png     — |adjoint − brute| vs |Δμ|; the window
#   plot_oracle_window.png        — the two above combined: curves on top,
#                                   relative deviation below on a shared μ
#                                   axis, with the T3 5% tolerance drawn
#                                   and the measured window shaded
#   plot_oracle_gscan_vs_joint.png — classical 1D g-scan vs joint (g,μ) path
#   plot_oracle_payoff.png        — the joint trajectory alone, sized for a
#                                   half-width slide column
#   plot_oracle_summary.png       — headline figure: truth / classical / joint
# ============================================================================

import csv
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Pin the SVG id salt: without it matplotlib draws ids from os.urandom
# and the SVG twin would differ on every run ([reproducible builds]).
matplotlib.rcParams["svg.hashsalt"] = "minlhc"


def _load_csv(path):
    with open(path) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        return {}
    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except ValueError:
                cols[k].append(v)
    return {k: np.asarray(v) if all(isinstance(x, float) for x in v) else v
            for k, v in cols.items()}


def plot_mu_scan(mu, L_brute, L_adj, mu0, mu_truth, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mu, L_brute, "b-", lw=2, label="Brute-force: rerun oracle at each μ")
    ax.plot(mu, L_adj,   "r--", lw=2,
            label="Adjoint: linearised oracle (one J extraction)")
    ax.axvline(mu0,       color="gray", ls=":", label=f"μ₀ (expansion)={mu0:.3f}")
    ax.axvline(mu_truth,  color="green", ls=":", label=f"μ★ (truth)={mu_truth:.3f}")
    ax.set_xlabel("μ  (lateral shower width; inside the opaque oracle)")
    ax.set_ylabel("log L (arbitrary zero)")
    ax.set_title("T3: Adjoint L(μ) vs brute-force L(μ)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_linearity_window(mu, L_brute, L_adj, mu0, out):
    dmu = mu - mu0
    err = np.abs(L_adj - L_brute) / (np.abs(L_brute) + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.abs(dmu), err, "ko-", markersize=4)
    ax.set_xlabel("|Δμ| from expansion point  μ₀")
    ax.set_ylabel("relative |L_adjoint − L_brute| / |L_brute|")
    ax.set_yscale("log")
    ax.set_title("Linearity window: where a single J extraction is adequate")
    ax.axhline(0.01, color="red", ls="--", alpha=0.7, label="1 % level")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_window_combined(mu, L_brute, L_adj, mu0, mu_truth, out, tol=0.05):
    """Curves + their relative deviation on a shared μ axis (the data/MC +
    ratio idiom). The window is recomputed here with the same outward-walk
    rule T3 uses, so figure and diagnostic cannot drift apart."""
    err = np.abs(L_adj - L_brute) / (np.abs(L_brute) + 1e-12)
    i0 = int(np.argmin(np.abs(mu - mu0)))
    lo = hi = i0
    while lo > 0 and err[lo - 1] < tol:
        lo -= 1
    while hi < len(mu) - 1 and err[hi + 1] < tol:
        hi += 1
    win_lo, win_hi = mu[lo], mu[hi]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.8), sharex=True,
                                   gridspec_kw={"height_ratios": [2.0, 1.3]})
    ax1.plot(mu, L_brute, "b-", lw=2.2,
             label="brute force: rerun simulator at each μ")
    ax1.plot(mu, L_adj, "r--", lw=2.2,
             label="linearized: cached events + frozen J (0 calls)")
    for ax in (ax1, ax2):
        ax.axvspan(win_lo, win_hi, color="green", alpha=0.12)
        ax.axvline(mu0, color="gray", ls=":", lw=1.5)
        ax.axvline(mu_truth, color="green", ls=":", lw=1.5)
        ax.grid(alpha=0.3)
    ax1.annotate(f"J built here (μ₀={mu0:.2f})", xy=(mu0, 0.97),
                 xycoords=("data", "axes fraction"), fontsize=10,
                 color="0.35", ha="left", xytext=(mu0 + 0.01, 0.92),
                 textcoords=("data", "axes fraction"))
    ax1.annotate(f"truth μ★={mu_truth:.2f}", xy=(mu_truth, 0.97),
                 xycoords=("data", "axes fraction"), fontsize=10,
                 color="green", ha="right", xytext=(mu_truth - 0.01, 0.92),
                 textcoords=("data", "axes fraction"))
    ax1.set_ylabel("score L (arbitrary zero)", fontsize=12)
    ax1.legend(fontsize=11, loc="lower left")
    ax1.set_title("The validity window: measured, not assumed (T3)",
                  fontsize=13)

    ax2.plot(mu, err, "ko-", ms=4)
    ax2.axhline(tol, color="red", ls="--", lw=1.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("relative deviation", fontsize=12)
    ax2.set_xlabel("μ (lateral shower width, inside the simulator)",
                   fontsize=12)
    ax2.text(mu[0] + 0.01, tol * 1.3, f"{tol:.0%} tolerance (T3)",
             color="red", fontsize=10)
    ax2.text((win_lo + win_hi) / 2, ax2.get_ylim()[0] * 1.6,
             f"window [{win_lo:.2f}, {win_hi:.2f}]", color="green",
             fontsize=10, ha="center")
    # The deviation dips again far to the right where the two curves cross
    # by accident — flag it on the figure so nobody reads it as validity.
    right = mu > win_hi + 0.1
    if right.any():
        j = int(np.argmin(np.where(right, err, np.inf)))
        ax2.annotate("accidental crossing —\nnot validity",
                     xy=(mu[j], err[j]), xytext=(mu[j] - 0.28, err[j] * 8),
                     fontsize=9, color="0.35",
                     arrowprops=dict(arrowstyle="->", color="0.35"))
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_gscan_vs_joint(gs_scan, Ls_scan, opt, truth_g, truth_mu,
                        classical_g, classical_L, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: classical 1D scan over g
    ax1.plot(gs_scan, Ls_scan, "b-o", markersize=3, label="Classical scan (μ fixed at nominal)")
    ax1.axvline(classical_g, color="red", ls="--",
                label=f"Classical best g = {classical_g:.3f}")
    ax1.axvline(truth_g, color="green", ls=":", label=f"Truth g = {truth_g:.3f}")
    ax1.set_xlabel("g"); ax1.set_ylabel("log L")
    ax1.set_title("Classical: scan over g, μ held at nominal")
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9)

    # Right: joint (g, μ) trajectory
    ax2.plot(opt["g"], opt["mu"], "k-", alpha=0.5)
    ax2.scatter(opt["g"], opt["mu"], c=opt["iter"], cmap="viridis",
                s=18, zorder=3, label="Joint optimizer path")
    ax2.scatter([classical_g], [1.0], marker="X", s=140, color="red",
                zorder=4, label="Classical scan endpoint (biased)")
    ax2.scatter([truth_g], [truth_mu], marker="*", s=220, color="green",
                zorder=5, label="Truth (g★, μ★)")
    # Mark Jacobian refreshes
    refreshes = opt["jacobian_refresh"]
    if len(refreshes):
        mask = refreshes > 0
        ax2.scatter(np.asarray(opt["g"])[mask], np.asarray(opt["mu"])[mask],
                    marker="o", s=80, facecolors="none", edgecolors="blue",
                    lw=1.5, label="J refresh")
    ax2.set_xlabel("g"); ax2.set_ylabel("μ")
    ax2.set_title("Joint (g, μ) optimizer via adjoint")
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=9)

    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_payoff_trajectory(opt, truth_g, truth_mu, nominal_g, nominal_mu,
                           classical_g, out):
    """The joint (g, μ) walk on its own, with fonts sized for a half-width
    slide column (the two-panel figure is illegible at that scale)."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(opt["g"], opt["mu"], "-", color="0.55", lw=1.5, zorder=2)
    ax.scatter(opt["g"], opt["mu"], c=opt["iter"], cmap="viridis", s=22,
               zorder=3, label="joint optimizer path")
    refreshes = np.asarray(opt["jacobian_refresh"])
    if len(refreshes):
        mask = refreshes > 0
        ax.scatter(np.asarray(opt["g"])[mask], np.asarray(opt["mu"])[mask],
                   marker="o", s=110, facecolors="none", edgecolors="blue",
                   lw=1.6, zorder=4, label="J refresh")
    ax.scatter([nominal_g], [nominal_mu], marker="s", s=130, color="0.35",
               zorder=5, label=f"start: nominal ({nominal_g:.2f}, "
                               f"{nominal_mu:.2f})")
    ax.scatter([classical_g], [nominal_mu], marker="X", s=170, color="red",
               zorder=5, label="classical scan endpoint (μ pinned)")
    ax.scatter([truth_g], [truth_mu], marker="*", s=300, color="green",
               zorder=6, label=f"truth ({truth_g:.2f}, {truth_mu:.2f})")
    ax.set_xlabel("g (electronics gain)", fontsize=13)
    ax.set_ylabel("μ (lateral shower width)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title("The joint fit walks to the buried knob", fontsize=13)
    ax.grid(alpha=0.3)
    # upper-right pocket between the classical X and the truth star:
    # the only region the trajectory never crosses (lower left holds
    # the convergence curl)
    ax.legend(fontsize=9.5, loc="upper right", bbox_to_anchor=(0.99, 0.90))
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    # SVG twin for the README (crisp on any display). Dropping the Date
    # metadata keeps re-runs bit-identical, matching the PNG determinism
    # documented in REPRODUCIBILITY.md.
    fig.savefig(out.replace(".png", ".svg"), metadata={"Date": None})
    plt.close(fig)


def plot_summary(summary, out):
    """Bar chart: truth / nominal / classical / joint for each parameter."""
    param_rows = {r["param"]: r for r in summary}
    cats = ["truth", "nominal_start", "classical", "joint"]
    labels = ["Truth", "Nominal start", "Classical scan", "Joint adjoint"]
    colors = ["green", "gray", "red", "blue"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, param in zip(axes, ("g", "mu")):
        row = param_rows[param]
        vals = [float(row[c]) for c in cats]
        ax.bar(labels, vals, color=colors, alpha=0.75)
        ax.set_ylabel(param)
        ax.set_title(f"Parameter: {param}")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        truth_v = float(row["truth"])
        ax.axhline(truth_v, color="green", ls=":", alpha=0.6)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Headline: classical (μ-blind) misses truth; joint adjoint recovers it")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    required = ["mu_scan.csv", "gscan_nominal_mu.csv", "opt_path.csv", "summary.csv"]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing CSVs:", ", ".join(missing), file=sys.stderr)
        print("Run ./opaque_oracle_demo first.", file=sys.stderr)
        sys.exit(1)

    mu_s    = _load_csv("mu_scan.csv")
    gscan   = _load_csv("gscan_nominal_mu.csv")
    opt     = _load_csv("opt_path.csv")
    summary = list(csv.DictReader(open("summary.csv")))
    sum_map = {r["param"]: r for r in summary}

    truth_g    = float(sum_map["g"]["truth"])
    truth_mu   = float(sum_map["mu"]["truth"])
    nominal_mu = float(sum_map["mu"]["nominal_start"])
    classical_g = float(sum_map["g"]["classical"])
    classical_L = float(sum_map["logL"]["classical"])

    plot_mu_scan(mu_s["mu"], mu_s["L_brute"], mu_s["L_adjoint"],
                 mu0=nominal_mu, mu_truth=truth_mu,
                 out="plot_oracle_mu_scan.png")
    plot_linearity_window(mu_s["mu"], mu_s["L_brute"], mu_s["L_adjoint"],
                          mu0=nominal_mu, out="plot_oracle_linearity.png")
    plot_window_combined(mu_s["mu"], mu_s["L_brute"], mu_s["L_adjoint"],
                         mu0=nominal_mu, mu_truth=truth_mu,
                         out="plot_oracle_window.png")
    plot_gscan_vs_joint(gscan["g"], gscan["logL"], opt,
                        truth_g=truth_g, truth_mu=truth_mu,
                        classical_g=classical_g, classical_L=classical_L,
                        out="plot_oracle_gscan_vs_joint.png")
    nominal_g = float(sum_map["g"]["nominal_start"])
    plot_payoff_trajectory(opt, truth_g=truth_g, truth_mu=truth_mu,
                           nominal_g=nominal_g, nominal_mu=nominal_mu,
                           classical_g=classical_g,
                           out="plot_oracle_payoff.png")
    plot_summary(summary, out="plot_oracle_summary.png")

    print("Wrote:")
    for p in ("plot_oracle_mu_scan.png", "plot_oracle_linearity.png",
              "plot_oracle_window.png", "plot_oracle_gscan_vs_joint.png",
              "plot_oracle_payoff.png", "plot_oracle_summary.png"):
        print(" ", p)


if __name__ == "__main__":
    main()
