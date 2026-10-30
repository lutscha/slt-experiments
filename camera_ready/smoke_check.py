"""Check the outputs of the --smoke runs. Run after `analyze.py --smoke`; exits non-zero on failure."""
import os

import numpy as np
import pandas as pd

out = f"{os.environ['RESULTS']}/camera_ready_out"
failures = []


def check(condition, message):
    print(("ok    " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)


def read(name, required=True):
    """Read an output CSV. Optional outputs (groups not run in this smoke test) are skipped if missing."""
    path = f"{out}/{name}"
    if required or os.path.exists(path):
        check(os.path.exists(path), f"{name} written")
    return pd.read_csv(path) if os.path.exists(path) else None


ln = read("layernorm.csv")
if ln is not None:
    check(set(ln["arch_id"]) == {"full_200", "full_lnh_200"}, "layernorm: both architectures ran")
    cols = ["sharpness", "alpha", "c_y", "q", "r_sym", "c_y_rest", "w_norm_sq"]
    check(np.isfinite(ln[cols].to_numpy()).all(), "layernorm: no NaN/inf in logged quantities")
    print(ln.groupby("arch_id")["sharpness"].agg(["min", "max"]).to_string())
    check((ln["sharpness"] > 0).all(), "layernorm: sharpness positive")
    check(((ln["q"] >= 0) & (ln["q"] <= 1 + 1e-4)).all(), "layernorm: 0 <= q <= 1 (eigenvector mass)")
    tol = 1e-3 * ln["c_y"].abs().max() + 1e-6
    check((ln["identity_residual"].abs() <= tol).all(), "layernorm: c_y = -2Sq + c_y_rest + r_sym holds")

ntk = read("ntk_sweep.csv", required=False)
if ntk is not None:
    check((ntk["lambda_max"] > 0).all(), "ntk: top NTK eigenvalue positive")
    check(ntk[["kta", "kta_centered", "kta_trace"]].abs().le(1 + 1e-6).all().all(), "ntk: |KTA| <= 1")
read("ntk_over_training.csv", required=False)
conv = read("ntk_convergence.csv", required=False)
if conv is not None:
    check(len(conv) >= 2, "ntk_convergence: several sample sizes")

adam = read("adamw.csv", required=False)
if adam is not None:
    check(np.isfinite(adam[["sharpness", "precond_sharpness"]].to_numpy()).all(), "adamw: no NaN/inf")
    check((adam["precond_sharpness"] > 0).all(), "adamw: preconditioned sharpness positive")

print()
print("SMOKE TEST FAILED:\n  " + "\n  ".join(failures) if failures else "SMOKE TEST PASSED")
raise SystemExit(1 if failures else 0)
