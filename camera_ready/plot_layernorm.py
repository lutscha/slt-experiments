"""LayerNorm figure of the paper appendix (layernorm_global_term.pdf); prints the decomposition table.

Usage: python camera_ready/plot_layernorm.py $RESULTS/camera_ready_out <output pdf>
"""
import csv
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
ETA = 0.02

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
})

runs = defaultdict(list)
for r in csv.DictReader(open(f"{src}/layernorm.csv")):
    runs[(r["arch_id"], float(r["wd"]))].append(r)
cols = ["step", "sharpness", "train_loss", "alpha", "c_y", "c_y_crit", "q", "c_y_rest", "r_sym", "S_w_norm_sq"]
data = {k: {c: np.array([float(r[c]) if r[c] else np.nan for r in rs]) for c in cols} for k, rs in runs.items()}

styles = [
    (("full_200", 0.05), "#c0392b", "-", r"MLP, $\gamma=0.05$"),
    (("full_lnh_200", 0.0), "#85c1e9", ":", r"LN MLP, $\gamma=0$"),
    (("full_lnh_200", 0.02), "#2e86c1", "--", r"LN MLP, $\gamma=0.02$"),
    (("full_lnh_200", 0.05), "#1b4f72", "-", r"LN MLP, $\gamma=0.05$"),
]

fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.2))
(a_s, a_l, a_n), (a_u, a_ln, a_d) = axes

for key, color, ls, label in styles:
    d = data[key]
    a_s.plot(d["step"], d["sharpness"], color=color, ls=ls, lw=0.7, label=label)
    a_l.plot(d["step"], d["train_loss"], color=color, ls=ls, lw=0.9, label=label)
    a_n.plot(d["step"], d["S_w_norm_sq"], color=color, ls=ls, lw=0.7, label=label)
a_s.axhline(2 / ETA, color="black", ls="--", lw=0.8)
a_s.text(10000, 2 / ETA - 10, r"$2/\eta$", ha="right", va="top", fontsize=7)
a_s.set_title("sharpness $S$")
a_l.set_title("training loss")
a_l.set_yscale("log")
a_n.set_title(r"$S\,\|\theta\|_2^2$")
a_n.set_yscale("log")
fig.legend(*a_s.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))


def crossing_panel(ax, key, title):
    d = data[key]
    m = d["alpha"] > 0  # as in Figure 6, steps with alpha <= 0 are not plotted
    ax.plot(d["step"][m], d["c_y"][m], color="#1b4f72" if "lnh" in key[0] else "#c0392b", lw=0.8, label=r"$c_y$")
    ax.plot(d["step"][m], d["c_y_crit"][m], color="black", ls="--", lw=0.8, label=r"$c_y^{\mathrm{crit}}$")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)


crossing_panel(a_u, ("full_200", 0.05), r"MLP, $\gamma=0.05$")
a_u.set_ylim(0, 60)
crossing_panel(a_ln, ("full_lnh_200", 0.05), r"LN MLP, $\gamma=0.05$")
a_ln.set_yscale("symlog", linthresh=10)

d = data[("full_lnh_200", 0.05)]
for y, color, ls, label in [
    (-2 * d["sharpness"] * d["q"], "#1b4f72", "-", r"$-2Sq$"),
    (d["c_y_rest"], "#e67e22", "-", r"$c_y^{\mathrm{rest}}$"),
    (d["r_sym"], "#7f8c8d", "-", r"$r_{\mathrm{sym}}$"),
    (d["c_y"], "black", "--", r"$c_y$"),
]:
    a_d.plot(d["step"], y, color=color, ls=ls, lw=0.7, label=label)
a_d.axhline(0, color="black", lw=0.4)
a_d.set_title(r"decomposition, LN MLP, $\gamma=0.05$")
a_d.set_ylim(-1100, 1900)
a_d.legend(loc="upper right", frameon=False, ncol=4, handlelength=1.0, columnspacing=0.6, handletextpad=0.3, fontsize=6.5)

for ax in axes.flat:
    ax.set_xlabel("step")
    ax.grid(True, which="major", ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out)

# Decomposition table: medians over training windows.
windows = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10001)]
for key, _, _, label in styles:
    d = data[key]
    for lo, hi in windows:
        if key[1] != 0.05 and lo < 6000:
            continue
        m = (d["step"] >= lo) & (d["step"] < hi)
        S, q = d["sharpness"][m], d["q"][m]
        vals = [np.median(S), np.median(q), np.median(-2 * S * q), np.median(d["c_y_rest"][m]),
                np.median(d["r_sym"][m]), np.median(d["c_y"][m]),
                np.median(d["c_y_crit"][m & (d["alpha"] > 0)]) if key[1] > 0 else np.nan]
        print(f"{label:28s} {lo:5d}-{min(hi, 10000):5d} " + " ".join(f"{v:10.4g}" for v in vals))
