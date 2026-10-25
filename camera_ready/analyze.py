"""Turn the camera-ready runs into CSVs and quick-look plots.

Usage: python camera_ready/analyze.py [--smoke]
Writes to $RESULTS/camera_ready_out/. Groups whose runs are missing are skipped.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run  # noqa: E402

# Parameter tensors whose layer feeds into a LayerNorm (weights and bias jointly scale invariant).
# For the unnormalized MLP the same hidden layers are used, so the decomposition is comparable.
INVARIANT_PREFIXES = {
    "full_200": ("1.", "4."),
    "full_lnh_200": ("1.", "4."),
    "full_ln_200": ("1.", "4.", "7."),
}


def load(directory, name):
    path = f"{directory}/{name}_final.pt"
    return torch.load(path) if os.path.exists(path) else None


def layernorm(out):
    rows, summary = [], []
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
    for e in run.LAYERNORM_RUNS:
        d = run.gd_dir(e["arch_id"], 0.02, e["wd"])
        eigs = load(d, "eigs")
        if eigs is None:
            print(f"missing: {d}")
            continue
        S = eigs[:, 0].numpy()
        steps = np.arange(len(S)) * run.EIG_FREQ["layernorm"]
        alpha, cy = load(d, "a").numpy(), load(d, "cy").numpy()
        loss = load(d, "train_loss").numpy()
        param_norms = load(d, "param_norms").numpy()
        cy_b, u2_b = load(d, "cy_blocks").numpy(), load(d, "u2_blocks").numpy()
        names = torch.load(f"{d}/block_names_final.pt")
        inv = np.array([n.startswith(INVARIANT_PREFIXES[e["arch_id"]]) for n in names])
        q = u2_b[:, inv].sum(1)
        r_sym = (cy_b[:, inv] + 2 * S[:, None] * u2_b[:, inv]).sum(1)
        cy_rest = cy_b[:, ~inv].sum(1)
        wd = e["wd"]
        df = pd.DataFrame({
            "arch_id": e["arch_id"], "wd": wd, "step": steps,
            "train_loss": loss[np.minimum(steps, len(loss) - 1)],
            "sharpness": S, "alpha": alpha, "c_y": cy,
            "c_y_crit": alpha / wd + wd if wd > 0 else np.nan,
            "c_y_over_S": cy / S,
            "w_norm_sq": param_norms[np.minimum(steps, len(param_norms) - 1)] ** 2,
            "q": q, "r_sym": r_sym, "c_y_rest": cy_rest,
            "identity_residual": cy - (-2 * S * q + cy_rest + r_sym),
        })
        df["S_w_norm_sq"] = df["sharpness"] * df["w_norm_sq"]
        rows.append(df)

        # c_y^crit ~ S^d during progressive sharpening (alpha > 0)
        fit = df[(df["alpha"] > 0) & (df["sharpness"] > 0)]
        d_exp = np.polyfit(np.log(fit["sharpness"]), np.log(fit["c_y_crit"]), 1)[0] if wd > 0 and len(fit) > 5 else np.nan
        summary.append({"arch_id": e["arch_id"], "wd": wd, "final_sharpness": float(S[-len(S) // 10:].mean()),
                        "final_loss": float(loss[-1]), "median_c_y_over_S": float(np.median(cy / S)),
                        "fit_exponent_d": float(d_exp), "max_abs_identity_residual": float(np.abs(df["identity_residual"]).max())})

        label = f"{e['arch_id']}, $\\gamma$={wd}"
        axes[0].plot(steps, S, label=label)
        axes[1].plot(np.arange(len(loss)), loss, label=label)
        axes[2].plot(steps, cy / S, label=label)
        if wd > 0:
            axes[3].plot(steps, cy, label=f"$c_y$ {label}")
            axes[3].plot(steps, df["c_y_crit"], "--", label=f"$c_y^{{crit}}$ {label}")
    if not rows:
        return
    for ax, title in zip(axes, ["sharpness", "train loss", "$c_y / S$", "$c_y$ vs $c_y^{crit}$"]):
        ax.set_title(title)
        ax.set_xlabel("step")
    axes[1].set_yscale("log")
    axes[3].set_yscale("symlog")
    axes[0].legend(fontsize=6)
    axes[3].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(f"{out}/layernorm.png", dpi=150)
    pd.concat(rows).to_csv(f"{out}/layernorm.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{out}/layernorm_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string())


def ntk(out):
    rows, over_training = [], []
    for e in run.NTK_RUNS:
        d = run.gd_dir("full_200", 0.02, e["wd"])
        path = f"{d}/ntk_final.pt"
        if not os.path.exists(path):
            print(f"missing: {d}")
            continue
        rec = torch.load(path)
        S = load(d, "eigs")[:, 0].numpy()
        for step, r in zip(rec["steps"], rec["records"]):
            over_training.append({"wd": e["wd"], "step": step, **r})
        rows.append({"wd": e["wd"], "steps": rec["steps"][-1],
                     "final_sharpness": float(S[-len(S) // 10:].mean()), **rec["records"][-1]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/ntk_sweep.csv", index=False)
    pd.DataFrame(over_training).to_csv(f"{out}/ntk_over_training.csv", index=False)
    print(df.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(df["wd"], df["lambda_max"], "o-", label=r"$\lambda_{max}(\hat\Theta)$")
    axes[0].plot(df["wd"], df["final_sharpness"], "s--", label="final sharpness")
    for col in ["kta", "kta_centered", "kta_trace"]:
        axes[1].plot(df["wd"], df[col], "o-", label=col)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/ntk_sweep.png", dpi=150)

    conv = f"{run.gd_dir('full_200', 0.02, run.NTK_RUNS[0]['wd'])}/ntk_convergence.pt"
    if os.path.exists(conv):
        pd.DataFrame(torch.load(conv)).to_csv(f"{out}/ntk_convergence.csv", index=False)


def adamw(out):
    rows = []
    fig, ax = plt.subplots(figsize=(6, 3.5))
    beta1, lr = 0.9, run.ADAMW_LR
    for e in run.ADAMW_RUNS:
        d = run.adamw_dir("full_200", lr, e["wd"], 5000)
        eigs = load(d, "eigs")
        if eigs is None:
            print(f"missing: {d}")
            continue
        steps = np.arange(len(eigs)) * run.EIG_FREQ["adamw"]
        threshold = (1 + beta1) * (2 - lr * e["wd"]) / (lr * (1 - beta1))
        df = pd.DataFrame({"wd": e["wd"], "step": steps, "sharpness": eigs[:, 0].numpy(),
                           "precond_sharpness": load(d, "peigs")[:, 0].numpy(), "threshold": threshold})
        df = df[df["step"] > 0]  # Adam has no preconditioner before the first step
        rows.append(df)
        line, = ax.plot(df["step"], df["precond_sharpness"], label=f"$\\gamma$={e['wd']}")
        ax.axhline(threshold, color=line.get_color(), linestyle="--", linewidth=0.8)
        loss = load(d, "train_loss")
        pd.DataFrame({"step": np.arange(len(loss)) * run.ADAMW_EVAL_FREQ, "train_loss": loss.numpy()}).to_csv(
            f"{out}/adamw_loss_wd{e['wd']}.csv", index=False)
    if not rows:
        return
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\lambda_{max}(P^{-1}H)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/adamw.png", dpi=150)
    pd.concat(rows).to_csv(f"{out}/adamw.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run.use_smoke_settings()
    out = f"{os.environ['RESULTS']}/camera_ready_out"
    os.makedirs(out, exist_ok=True)
    for f in (layernorm, ntk, adamw):
        print(f"== {f.__name__}")
        f(out)
    print(json.dumps({"written_to": out}))
