"""AdamW figure of the paper appendix (adamw_precond_sharpness.pdf).

Usage: python camera_ready/plot_adamw.py $RESULTS/camera_ready_out <output pdf>
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
ETA = 1e-4

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
})

sharp = defaultdict(lambda: ([], []))
for r in csv.DictReader(open(f"{src}/adamw.csv")):
    s, p = sharp[float(r["wd"])]
    s.append(int(r["step"]))
    p.append(float(r["precond_sharpness"]))

gammas = [0.0, 1.0, 4.0, 10.0]
colors = ["#1b4f72", "#2e86c1", "#e67e22", "#c0392b"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
for g, c in zip(gammas, colors):
    steps, p = sharp[g]
    ax1.plot(steps, [x / 1e3 for x in p], color=c, lw=0.9, label=fr"$\gamma={g:g}$")
    loss = list(csv.DictReader(open(f"{src}/adamw_loss_wd{g}.csv")))
    ax2.plot([int(r["step"]) for r in loss], [float(r["train_loss"]) for r in loss], color=c, lw=0.9,
             label=fr"$\gamma={g:g}$")

ax1.axhline(38 / ETA / 1e3, color="black", ls="--", lw=1.0, label=r"$\frac{38}{\eta}$")
ax1.set_ylim(200, 410)
ax1.set_xlabel("step")
ax1.set_ylabel(r"$\lambda_{\max}(P^{-1}H)\ (\times 10^3)$")
ax1.set_title("preconditioned sharpness")
ax1.legend(ncol=2, loc="lower right", frameon=False)

ax2.set_yscale("log")
ax2.set_xlabel("step")
ax2.set_ylabel("training loss")
ax2.set_title("training loss")

for ax in (ax1, ax2):
    ax.grid(True, which="major", ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(out)
