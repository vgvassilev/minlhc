#!/usr/bin/env python3
# The T6 fix, shown rather than printed: both objectives scanned on
# shared events along the truth axes (score_scans.csv, written by
# bias_check). The per-event score peaks off truth; the mean-level
# score peaks exactly at it. Run from the repository root.
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("score_scans.csv")))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for ax, axis, truth, xlabel in ((axes[0], "g", 1.00,
                                 "g (electronics gain), μ held at truth"),
                                (axes[1], "mu", 0.85,
                                 "μ (shower width), g held at truth")):
    x = [float(r["x"]) for r in rows if r["axis"] == axis]
    for col, label, color in (("L_event",
                               "per-event score (variance penalty)",
                               "tab:red"),
                              ("L_mean", "mean-level score (the T6 fix)",
                               "tab:green")):
        y = [float(r[col]) for r in rows if r["axis"] == axis]
        top = max(y)
        span = top - min(y)
        yn = [(v - top) / span for v in y]
        ax.plot(x, yn, color=color, lw=2, label=label)
        xb = x[y.index(top)]
        ax.plot([xb], [0], "o", color=color, ms=7)
        ax.annotate(f"{xb:.3f}", (xb, 0), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=10, color=color)
    ax.axvline(truth, color="0.3", ls="--", lw=1.2)
    ax.text(truth + 0.004, -0.97, f"truth {truth:.2f}", ha="left",
            fontsize=10, color="0.3")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylim(-1.08, 0.14)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("score, shifted and scaled (peak = 0)", fontsize=11)
axes[0].legend(fontsize=9.5, loc="lower left")
fig.suptitle("Optimizer removed: where each objective puts its optimum",
             fontsize=13)
fig.tight_layout()
fig.savefig("plot_score_fix.png", dpi=140)
print("wrote plot_score_fix.png")
