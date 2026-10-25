"""Experiments for the NeurIPS 2026 camera-ready.

Usage (from the repo root, with DATASETS and RESULTS set):
    python camera_ready/run.py layernorm        # R3-R5: LayerNorm vs. unnormalized MLP, symmetry decomposition
    python camera_ready/run.py ntk              # R1: NTK top eigenvalue and KTA across weight decay
    python camera_ready/run.py ntk_convergence  # R1: NTK sample-size check (needs the ntk group first)
    python camera_ready/run.py adamw            # R2: full-batch AdamW, preconditioned sharpness
    python camera_ready/run.py <group> --only 0 # run a single entry of a group
    python camera_ready/run.py <group> --smoke  # tiny CPU-sized version, to check that everything runs

Finished runs are skipped, so a group can be restarted after a Colab disconnect.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

DATASET = "cifar10-5k"
SEED = 0
# How often sharpness is measured (and loss logged, for AdamW); smaller in --smoke mode.
EIG_FREQ = {"layernorm": 25, "ntk": 100, "adamw": 50}
ADAMW_EVAL_FREQ = 25

# R3-R5. eta = 0.02 as in the paper's MLP experiments; gamma = 0.05 is the rebuttal setting.
LAYERNORM_RUNS = [
    dict(arch_id="full_200", wd=0.05),
    dict(arch_id="full_ln_200", wd=0.05),
    dict(arch_id="full_lnh_200", wd=0.05),
    dict(arch_id="full_ln_200", wd=0.0),
    dict(arch_id="full_ln_200", wd=0.02),
]

# R1. The gamma grid of the NTK eigenvalue figure; larger gamma needs more steps to stabilize.
NTK_RUNS = [
    dict(wd=1e-5, max_steps=6000), dict(wd=5e-5, max_steps=6000), dict(wd=1e-4, max_steps=6000),
    dict(wd=5e-4, max_steps=6000), dict(wd=1e-3, max_steps=6000), dict(wd=5e-3, max_steps=8000),
    dict(wd=1e-2, max_steps=12000), dict(wd=1.5e-2, max_steps=15000), dict(wd=2e-2, max_steps=20000),
    dict(wd=2.5e-2, max_steps=25000),
]

# R2. eta * gamma per step spans the same range as the GD runs (eta * gamma = 4e-4 for eta = gamma = 0.02).
ADAMW_LR = 1e-4
ADAMW_RUNS = [dict(wd=0.0), dict(wd=1.0), dict(wd=4.0), dict(wd=10.0)]


def use_smoke_settings():
    global DATASET, ADAMW_EVAL_FREQ
    DATASET = "cifar10-1k"
    EIG_FREQ.update(layernorm=10, ntk=10, adamw=10)
    ADAMW_EVAL_FREQ = 10


def gd_dir(arch_id, lr, wd):
    from utilities import get_gd_directory
    return get_gd_directory(DATASET, lr, arch_id, SEED, "gd", "mse", wd)


def adamw_dir(arch_id, lr, wd, batch_size):
    return (f"{os.environ['RESULTS']}/{DATASET}/{arch_id}/seed_{SEED}/mse/adamw/"
            f"lr_{lr}_beta1_0.9_beta2_0.999_eps_1e-07/{batch_size}/{wd}")


def finished(directory):
    return os.path.exists(f"{directory}/eigs_final.pt")


def run_layernorm(entries, smoke):
    from gd import main
    for e in entries:
        lr = 0.02
        if finished(gd_dir(e["arch_id"], lr, e["wd"])):
            print(f"skip (done): {e}")
            continue
        print(f"=== layernorm: {e}")
        main(dataset=DATASET, arch_id=e["arch_id"], loss_type="mse", opt="gd", lr=lr, wd=e["wd"],
             max_steps=40 if smoke else 10000, neigs=1, eig_freq=EIG_FREQ["layernorm"],
             abridged_size=5000, seed=SEED, record_norms=True, track_cy=True, record_blocks=True,
             save_freq=-1 if smoke else 1000, save_model=True)


def run_ntk(entries, smoke):
    from gd import main
    for e in entries:
        lr = 0.02
        if finished(gd_dir("full_200", lr, e["wd"])):
            print(f"skip (done): {e}")
            continue
        print(f"=== ntk: {e}")
        main(dataset=DATASET, arch_id="full_200", loss_type="mse", opt="gd", lr=lr, wd=e["wd"],
             max_steps=40 if smoke else e["max_steps"], neigs=1, eig_freq=EIG_FREQ["ntk"],
             abridged_size=5000, seed=SEED, record_norms=True, track_cy=False,
             ntk_freq=20 if smoke else 1000, ntk_final=True, abridged_size_ntk=20 if smoke else 500,
             save_freq=-1 if smoke else 2000, save_model=True)


def run_ntk_convergence(smoke):
    """NTK top eigenvalue and KTA against the number of samples n, at the final weights of the smallest gamma."""
    from archs import load_architecture
    from data import load_dataset, take_first
    from ntk import ntk_metrics
    wd = NTK_RUNS[0]["wd"]
    directory = gd_dir("full_200", 0.02, wd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_architecture("full_200", DATASET).to(device)
    network.load_state_dict(torch.load(f"{directory}/snapshot_final", map_location=device))
    train, _ = load_dataset(DATASET, "mse")
    records = []
    for n in ([10, 20] if smoke else [100, 250, 500, 750, 1000]):
        records.append(ntk_metrics(network, take_first(train, n)))
        print(records[-1])
    torch.save(records, f"{directory}/ntk_convergence.pt")


def run_adamw(entries, smoke):
    from adam import main
    batch_size = 5000  # full batch
    for e in entries:
        if finished(adamw_dir("full_200", ADAMW_LR, e["wd"], batch_size)):
            print(f"skip (done): {e}")
            continue
        print(f"=== adamw: {e}")
        main(dataset=DATASET, arch_id="full_200", loss="mse", opt="adam", lr=ADAMW_LR,
             beta1=0.9, beta2=0.999, epsilon=1e-7, max_steps=40 if smoke else 10000, neigs=1,
             eig_freq=EIG_FREQ["adamw"], eval_freq=ADAMW_EVAL_FREQ, batch_size=batch_size,
             abridged_size=5000, seed=SEED, wd=e["wd"], adamw=True, decay_all=True,
             save_freq=-1 if smoke else 1000, save_model=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["layernorm", "ntk", "ntk_convergence", "adamw"])
    parser.add_argument("--only", type=int, default=None, help="run only this entry of the group")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check that the code works")
    args = parser.parse_args()

    if args.smoke:
        use_smoke_settings()

    def select(entries):
        return entries if args.only is None else [entries[args.only]]

    if args.group == "layernorm":
        run_layernorm(select(LAYERNORM_RUNS), args.smoke)
    elif args.group == "ntk":
        run_ntk(select(NTK_RUNS), args.smoke)
    elif args.group == "ntk_convergence":
        run_ntk_convergence(args.smoke)
    elif args.group == "adamw":
        run_adamw(select(ADAMW_RUNS), args.smoke)
