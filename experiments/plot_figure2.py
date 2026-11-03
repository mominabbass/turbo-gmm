"""
usage: python experiments/plot_figure2.py [--out figure2.pdf]
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# The ladder, in order. Labels are what the paper calls them.
RUNGS = ["A1 torch lifted mat.", "A2 +fused score kernel",
         "A3 fused two-pass", "A4 fused one-pass"]
SHORT = {"A1 torch lifted mat.":   "Alg. 1\n(materialize)",
         "A2 +fused score kernel": "+fused\nscore",
         "A3 fused two-pass":      "+fuse E/M\n(Alg. 3)",
         "A4 fused one-pass":      "+one pass\n(Alg. 2)"}
COLOR = {"A1 torch lifted mat.":   "#7f7f7f",   # baseline, grey
         "A2 +fused score kernel": "#9467bd",
         "A3 fused two-pass":      "#1f77b4",
         "A4 fused one-pass":      "#d62728"}    # ours, tab:red, as in Figure 1

FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 12, 10, 9, 12
GRID_COLOR = "#b0b0b0"
FIG_W, FIG_H = 10.06, 3.30

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--latency", default=os.path.join(HERE, "results/ablation.jsonl"))
ap.add_argument("--out", default=os.path.join(HERE, "figure2.pdf"))
args = ap.parse_args()


def tick(N, K, d):
    n = f"{N // 1_000_000}M" if N >= 1_000_000 else f"{N // 1000}K"
    return f"N={n}\nK={K}\nd={d}"


rows = [json.loads(l) for l in open(args.latency) if l.strip()]
shapes = []
for r in rows:
    s = (r["N"], r["K"], r["d"])
    if s not in shapes:
        shapes.append(s)
by = {(r["N"], r["K"], r["d"], r["rung"]): r for r in rows}

fig, axL = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))

# ------------------------------------------------------------------ left panel
w = 0.8 / len(RUNGS)
missing = []
for j, rung in enumerate(RUNGS):
    xs, ys = [], []
    for i, sh in enumerate(shapes):
        r = by.get((*sh, rung))
        base = by.get((*sh, RUNGS[0]))
        x = i + (j - (len(RUNGS) - 1) / 2) * w
        if r is None or r.get("ms") is None or base is None or base.get("ms") is None:
            if r is not None and r.get("ms") is None:
                missing.append(x)
            continue
        xs.append(x)
        ys.append(base["ms"] / r["ms"])          # speedup over the Alg. 1 dataflow
    if xs:
        axL.bar(xs, ys, width=w * 0.92, label=SHORT[rung], color=COLOR[rung],
                edgecolor="black", linewidth=0.3)
axL.set_yscale("log")
axL.axhline(1.0, color="black", lw=0.8, ls="--", zorder=1)
lo, hi = axL.get_ylim()
axL.set_ylim(min(lo, 0.03), hi * 1.6)
for x in missing:
    axL.plot(x, min(lo, 0.03) * 1.35, "x", color="red", ms=6, mew=1.6,
             zorder=5, clip_on=False)
axL.set_xticks(range(len(shapes)))
axL.set_xticklabels([tick(*s) for s in shapes], fontsize=FS_TICK)
axL.tick_params(axis="y", labelsize=FS_TICK)
axL.set_xlim(-0.6, len(shapes) - 0.4)
axL.set_ylabel("Speedup over Alg. 1 dataflow", fontsize=FS_LABEL)
axL.set_title("Cumulative mechanism ablation: from the Algorithm 1 dataflow to "
              "Algorithm 2", fontsize=FS_TITLE, fontweight="bold")
axL.grid(axis="y", ls=":", lw=0.5, color=GRID_COLOR)
axL.set_axisbelow(True)

h, l = axL.get_legend_handles_labels()
h.append(Line2D([], [], ls="none", marker="x", color="red", ms=6, mew=1.6))
l.append("exceeds SRAM")
fig.tight_layout(rect=(0.005, 0.005, 0.995, 0.86))
fig.legend(h, l, loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.01), fontsize=FS_LEG,
           columnspacing=1.1, handletextpad=0.5)
fig.savefig(args.out)
fig.savefig(args.out.replace(".pdf", ".png"), dpi=220)
print("wrote", args.out)

# ------------------------------------------------------------------ the table
print("\n" + "=" * 96)
print("PER-RUNG FACTORS (each rung relative to the one above it)")
print("=" * 96)
print(f"{'shape':22s} " + " ".join(f"{r.split()[0]:>9s}" for r in RUNGS)
      + f" {'A1->A4':>9s}")
for sh in shapes:
    ms = [by.get((*sh, r), {}).get("ms") for r in RUNGS]
    fac = []
    for a, b in zip(ms, ms[1:]):
        fac.append(f"{a/b:8.2f}x" if (a and b) else "        -")
    tot = f"{ms[0]/ms[-1]:8.2f}x" if (ms[0] and ms[-1]) else "        -"
    lab = f"N={sh[0]//1000}K K={sh[1]} d={sh[2]}"
    print(f"{lab:22s} {'  base':>9s} " + " ".join(fac) + f" {tot:>9s}")
