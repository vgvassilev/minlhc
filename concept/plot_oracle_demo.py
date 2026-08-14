#!/usr/bin/env python3
# ============================================================================
# Plots for concept/opaque_oracle_demo.cpp
#
# Run AFTER the C++ demo has written mu_scan.csv, gscan_nominal_mu.csv,
# opt_path.csv, and summary.csv to the current working directory.
#
# Produces four PNGs that together make the pedagogical argument:
#
#   plot_oracle_mu_scan.png       — adjoint L(μ) vs brute-force L(μ)
#   plot_oracle_linearity.png     — |adjoint − brute| vs |Δμ|; the window
#   plot_oracle_gscan_vs_joint.png — classical 1D g-scan vs joint (g,μ) path
#   plot_oracle_summary.png       — headline figure: truth / classical / joint
# ============================================================================

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


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
    plot_gscan_vs_joint(gscan["g"], gscan["logL"], opt,
                        truth_g=truth_g, truth_mu=truth_mu,
                        classical_g=classical_g, classical_L=classical_L,
                        out="plot_oracle_gscan_vs_joint.png")
    plot_summary(summary, out="plot_oracle_summary.png")

    print("Wrote:")
    for p in ("plot_oracle_mu_scan.png", "plot_oracle_linearity.png",
              "plot_oracle_gscan_vs_joint.png", "plot_oracle_summary.png"):
        print(" ", p)


if __name__ == "__main__":
    main()
