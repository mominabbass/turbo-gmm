"""
usage: python experiments/plot_figure1.py [--out figure1.pdf]
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
# TrueType, not Type 3: matplotlib's PDF default is Type 3
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ARM_ORDER = ["turbo-gmm", "flash-gmm", "cuvs", "pytorch", "sklearn",
             "gmmx", "torchgmm", "pomegranate"]
LABEL = {"turbo-gmm": "Turbo-GMM (ours)", "flash-gmm": "Flash-GMM",
         "cuvs": "cuVS", "pytorch": "PyTorch", "sklearn": "scikit-learn",
         "gmmx": "gmmx (JAX)", "torchgmm": "torchgmm", "pomegranate": "pomegranate"}
COLOR = {"turbo-gmm": "#d62728",      # tab:red   -- ours first, as flash-kmeans does
         "flash-gmm": "#1f77b4",  # tab:blue  -- the reference competitor
         "cuvs": "#ff7f0e",           # tab:orange
         "pytorch": "#2ca02c",        # tab:green
         "sklearn": "#9467bd",        # tab:purple
         "gmmx": "#8c564b",           # tab:brown
         "torchgmm": "#e377c2",       # tab:pink
         "pomegranate": "#7f7f7f"}    # tab:gray

FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 12, 10, 9, 12
GRID_COLOR = "#b0b0b0"

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(HERE, "results/results.jsonl"))
ap.add_argument("--flash",
                default=os.path.join(HERE, "results/fig1_matched_tf32x3.jsonl"))
ap.add_argument("--out", default=os.path.join(HERE, "figure1.pdf"))
args = ap.parse_args()

ext = [json.loads(l) for l in open(args.results) if l.strip()]
fla = [json.loads(l) for l in open(args.flash) if l.strip()]

rows = []
for r in ext:
    if r["arm"] in ("turbo-gmm",):        # superseded by the re-measured file
        continue
    if r.get("status") == "ok" and r.get("all_ms"):
        r["ms_per_iter"] = min(r["all_ms"]) / r["iters"]
    r["diverged"] = (r.get("status") == "ok"
                     and not math.isfinite(r.get("final_ll", float("nan"))))
    r["good"] = r.get("status") == "ok" and not r["diverged"]
    rows.append(r)

# ---- our method
PANEL_OF = {}
for r in ext:
    PANEL_OF.setdefault((r["N"], r["K"], r["d"]), set()).add(r["panel"])
FL = {(r["N"], r["K"], r["d"], r["arm"]): r for r in fla}
alg3_used = set()


def emit(rec, arm):
    rec = dict(rec); rec["arm"] = arm
    ws = rec.get("weight_sum")
    rec["diverged"] = (rec.get("status") == "ok"
                       and (not math.isfinite(rec.get("final_ll", float("nan")))
                            or (ws is not None and abs(ws - 1.0) > 1e-2)))
    rec["good"] = rec.get("status") == "ok" and not rec["diverged"]
    for p in PANEL_OF.get((rec["N"], rec["K"], rec["d"]), {rec.get("panel")}):
        q = dict(rec); q["panel"] = p
        rows.append(q)


for (N, K, d, arm) in list(FL):
    if arm != "turbo-gmm-alg2":
        continue
    a2, a3 = FL[(N, K, d, "turbo-gmm-alg2")], FL.get((N, K, d, "turbo-gmm-alg3"))
    if a2.get("status") == "ok":
        emit(a2, "turbo-gmm")
    elif a3 is not None:
        alg3_used.add((N, K, d))
        emit(a3, "turbo-gmm")
for (N, K, d, arm), r in FL.items():
    if arm == "flash-gmm":
        emit(r, "flash-gmm")

panels = []
for r in ext:
    if r["panel"] not in panels:
        panels.append(r["panel"])
by = {(r["panel"], r["N"], r["K"], r["d"], r["arm"]): r for r in rows}


def configs_of(panel):
    seen = []
    for r in ext:
        key = (r["N"], r["K"], r["d"])
        if r["panel"] == panel and key not in seen:
            seen.append(key)
    return seen


def tick(N, K, d):
    n = f"{N // 1_000_000}M" if N >= 1_000_000 else f"{N // 1000}K"
    return f"N={n}\nK={K}\nd={d}"


def pretty(panel):
    return panel.replace("crossover", "Crossover")

FIG_W, FIG_H = 10.06, 6.20
nrow = 2
ncol = int(math.ceil(len(panels) / nrow))
fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W, FIG_H))
axes = np.atleast_1d(axes).ravel()

for ax, panel in zip(axes, panels):
    cfgs = configs_of(panel)
    arms = [a for a in ARM_ORDER
            if any((panel, N, K, d, a) in by for (N, K, d) in cfgs)]
    w = 0.8 / max(len(arms), 1)
    oom, div, stars = [], [], []
    for j, arm in enumerate(arms):
        xs, ys = [], []
        for i, (N, K, d) in enumerate(cfgs):
            r = by.get((panel, N, K, d, arm))
            x = i + (j - (len(arms) - 1) / 2) * w
            if r is None:
                continue
            if r["good"]:
                xs.append(x); ys.append(r["ms_per_iter"])
                if arm == "turbo-gmm" and (N, K, d) in alg3_used:
                    stars.append((x, r["ms_per_iter"]))
            elif r["diverged"]:
                div.append(x)
            else:
                oom.append(x)
        if xs:
            ax.bar(xs, ys, width=w * 0.92, label=LABEL[arm], color=COLOR[arm],
                   edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 2.2)                 # headroom for the stars
    for x in oom:
        ax.plot(x, lo * 1.30, "x", color="red", ms=6, mew=1.6, zorder=5, clip_on=False)
    for x in div:
        ax.plot(x, lo * 1.30, "o", mfc="none", mec="red", ms=5, mew=1.4, zorder=5,
                clip_on=False)
    for (x, y) in stars:
        ax.plot(x, y * 1.55, marker="*", color="black", ms=7, ls="none", zorder=6,
                clip_on=False)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([tick(*c) for c in cfgs], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlim(-0.6, len(cfgs) - 0.4)
    ax.set_title(pretty(panel), fontsize=FS_TITLE, fontweight="bold")
    ax.set_ylabel("Latency (ms) - Log Scale", fontsize=FS_LABEL)
    ax.grid(axis="y", ls=":", lw=0.5, color=GRID_COLOR)
    ax.set_axisbelow(True)

for ax in axes[len(panels):]:
    ax.axis("off")

h, l = [], []
for ax in axes[:len(panels)]:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
h.append(Line2D([], [], ls="none", marker="*", color="black", ms=7))
l.append("Alg. 3 (Alg. 2 exceeds SRAM)")
h.append(Line2D([], [], ls="none", marker="x", color="red", ms=6, mew=1.6))
l.append("OOM / unsupported")
h.append(Line2D([], [], ls="none", marker="o", mfc="none", mec="red", ms=5, mew=1.4))
l.append("no valid mixture")
fig.tight_layout(rect=(0.005, 0.005, 0.995, 0.872))

fig.legend(h, l, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.005), fontsize=FS_LEG,
           columnspacing=1.1, handletextpad=0.5)
fig.savefig(args.out)
fig.savefig(args.out.replace(".pdf", ".png"), dpi=220)
print("wrote", args.out)
print("Algorithm 3 used at:", sorted(alg3_used) or "none")

# ------------------------------------------------------------------ quality table
print("\n" + "=" * 120)
print("PAIRED TABLE -- latency is only meaningful next to the objective it reached")
print("D = completed but non-finite log-likelihood (excluded from speedups); "
      "* = our Alg. 3 fallback")
print("=" * 120)
print(f"{'panel':18s} {'N':>10s} {'K':>5s} {'d':>4s} {'arm':17s} "
      f"{'ms/iter':>10s} {'speedup':>8s} {'final LL':>16s} {'ARI':>7s} {'peak MB':>8s}")
for panel in panels:
    for (N, K, d) in configs_of(panel):
        f = by.get((panel, N, K, d, "turbo-gmm"))
        for arm in ARM_ORDER:
            r = by.get((panel, N, K, d, arm))
            if not r:
                continue
            if r.get("status") != "ok":
                print(f"{panel:18s} {N:10,d} {K:5d} {d:4d} {arm:17s} {r['status']:>10s}")
                continue
            flag = " D" if r["diverged"] else ""
            if arm == "turbo-gmm" and (N, K, d) in alg3_used:
                flag += " *"
            sp = (r["ms_per_iter"] / f["ms_per_iter"]) if (f and f["good"]) else float("nan")
            pk = r.get("peak_mb")
            print(f"{panel:18s} {N:10,d} {K:5d} {d:4d} {arm:17s} "
                  f"{r['ms_per_iter']:10.2f} {sp:7.2f}x {r['final_ll']:16.8f} "
                  f"{r.get('ari', float('nan')):7.4f} "
                  f"{(f'{pk:.0f}' if pk else '-'):>8s}{flag}")
