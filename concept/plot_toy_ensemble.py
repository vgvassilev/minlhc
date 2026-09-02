#!/usr/bin/env python3
# Single-panel ensemble evidence figure for the RQ4 result: 50 fits as a
# tight cluster, the planted truth far away, the gap annotated in sigmas
# of the per-fit error, and toy 0 (the single fit shown on the payoff
# figure) highlighted. Input: toy_ensemble.csv, written by toy_ensemble
# (see concept/toy_ensemble.cpp). Run from the repository root.
import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
# Pinned SVG id salt for bit-identical re-runs (ids otherwise come from
# os.urandom); pairs with the Date-free metadata at savefig below.
matplotlib.rcParams["svg.hashsalt"] = "minlhc"
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("toy_ensemble.csv")))
g = [float(r["g_fit"]) for r in rows]
m = [float(r["mu_fit"]) for r in rows]
eg = [float(r["g_err"]) for r in rows]
em = [float(r["mu_err"]) for r in rows]
gm, mm = st.mean(g), st.mean(m)
egm, emm = st.mean(eg), st.mean(em)

fig, ax = plt.subplots(figsize=(6.8, 5.4))
ax.scatter(g, m, s=26, color="0.15", alpha=0.65, zorder=3,
           label=f"{len(rows)} independent pseudo-experiments")
ax.scatter([g[0]], [m[0]], s=150, facecolors="none", edgecolors="tab:orange",
           lw=2.0, zorder=4, label="toy 0 — the fit on the payoff figure")
ax.errorbar([gm], [mm], xerr=[egm], yerr=[emm], fmt="s", ms=6,
            color="tab:blue", capsize=4, zorder=5,
            label="ensemble mean ± per-fit Fisher error")
ax.scatter([1.00], [0.85], marker="*", s=340, color="green", zorder=6,
           label="planted truth (1.00, 0.85)")
ax.annotate("", xy=(gm + 0.004, mm + 0.004), xytext=(0.997, 0.847),
            arrowprops=dict(arrowstyle="->", color="0.4", ls="--", lw=1.4))
ax.text(0.968, 0.838,
        f"the gap: {gm - 1.00:+.3f} in g, {mm - 0.85:+.3f} in μ\n"
        f"= {abs(gm - 1.00) / egm:.0f}σ and {abs(mm - 0.85) / emm:.0f}σ "
        "of the per-fit error", fontsize=10.5, color="0.25", ha="center")
ax.annotate("all fits land here —\ntight (precise), wrong (biased)",
            xy=(gm - 0.003, mm + 0.004), xytext=(0.928, 0.845),
            fontsize=10.5, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.4", lw=1.2))
ax.set_xlim(0.920, 1.018)
ax.set_ylim(0.790, 0.868)
ax.set_xlabel("fitted g (electronics gain)", fontsize=13)
ax.set_ylabel("fitted μ (lateral shower width)", fontsize=13)
ax.tick_params(labelsize=11)
ax.set_title("50 repeats of the experiment, fresh randomness each time",
             fontsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=9.5, loc="lower right")
fig.tight_layout()
fig.savefig("plot_toy_ensemble.png", dpi=140)
# SVG twin for the README; Date metadata dropped so re-runs stay
# bit-identical (see the determinism note in REPRODUCIBILITY.md).
fig.savefig("plot_toy_ensemble.svg", metadata={"Date": None})
print("wrote plot_toy_ensemble.png and plot_toy_ensemble.svg")
