# transformer_gd.py
import os
import torch
import argparse
from torch.nn.utils import parameters_to_vector

# --- your utilities ---
from ..utilities import (
    get_gd_directory,
    get_gd_optimizer,
    get_hessian_eigenvalues,
    save_files,
    save_files_final,
)

# --- Transformer LM + dataset loader ---
from .lm_model import TransformerLM, generate_square_subsequent_mask
from .lm_dataset import load_wikitext2_lm, get_batch


# ======================================================================
# Full-batch GD training for Transformer LM on WikiText-2
# ======================================================================
def main(
    lr: float,
    max_steps: int,
    bptt: int = 35,
    batch_size: int = 20,
    seed: int = 0,
    opt: str = "gd",
    wd: float = 0.0,
    neigs: int = 0,
    eig_freq: int = -1,
    save_freq: int = -1,
    nproj: int = 0,
    iterate_freq: int = -1,
    save_model: bool = False,
    dataset: str = "wikitext2",
):

    # ----------------------------------------------------------
    # Prepare output dir
    # ----------------------------------------------------------
    arch_id = "transformer"
    loss_type = "nll"

    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss_type, wd)
    os.makedirs(directory, exist_ok=True)
    print(f"\nSaving results to:\n  {directory}\n")

    # ----------------------------------------------------------
    # Seed
    # ----------------------------------------------------------
    torch.manual_seed(seed)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    train_data, valid_data, test_data, vocab = load_wikitext2_lm(
        bptt=bptt, batch_size=batch_size
    )

    ntoken = len(vocab)
    print(f"Vocab size = {ntoken}")
    print(f"Train tokens = {train_data.numel()}")
    print(f"Test  tokens = {test_data.numel()}")

    # ----------------------------------------------------------
    # Initialize model
    # ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerLM(
        ntoken=ntoken,
        ninp=200,
        nhead=2,
        nhid=200,
        nlayers=2,
        dropout=0.0,
    ).to(device)

    # ----------------------------------------------------------
    # Static causal mask
    # ----------------------------------------------------------
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    # ----------------------------------------------------------
    # Loss
    # ----------------------------------------------------------
    loss_fn = torch.nn.NLLLoss(reduction="sum")

    # ----------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------
    optimizer = get_gd_optimizer(model.parameters(), opt, lr, momentum=0.0, wd=wd)

    # ----------------------------------------------------------
    # Logs
    # ----------------------------------------------------------
    train_loss_log = torch.zeros(max_steps)
    test_loss_log = torch.zeros(max_steps)

    eigs = torch.zeros(max_steps // eig_freq if eig_freq > 0 else 0, neigs)

    if nproj > 0:
        iterates = torch.zeros(max_steps // iterate_freq, nproj)
        proj_matrix = torch.randn(nproj, len(parameters_to_vector(model.parameters())))
    else:
        iterates = None

    # ==========================================================
    # Full-batch GD loop
    # ==========================================================
    for step in range(max_steps):

        # ===========================================
        # Evaluate train loss
        # ===========================================
        model.eval()
        total_train_loss = 0.0

        with torch.no_grad():
            for i in range(0, train_data.size(0) - 1, bptt):
                data, targets = get_batch(train_data, i, bptt)
                data = data.to(device)
                targets = targets.to(device)

                output = model(data, src_mask)
                output = output.reshape(-1, ntoken)
                targets = targets.reshape(-1)

                total_train_loss += loss_fn(output, targets).item()

        train_loss_log[step] = total_train_loss / train_data.size(0)

        # ===========================================
        # Evaluate test loss
        # ===========================================
        total_test_loss = 0.0
        with torch.no_grad():
            for i in range(0, test_data.size(0) - 1, bptt):
                data, targets = get_batch(test_data, i, bptt)
                data = data.to(device)
                targets = targets.to(device)

                output = model(data, src_mask)
                output = output.reshape(-1, ntoken)
                targets = targets.reshape(-1)

                total_test_loss += loss_fn(output, targets).item()

        test_loss_log[step] = total_test_loss / test_data.size(0)

        print(
            f"{step}\tTrain NLL={train_loss_log[step]:.4f}\t"
            f"Test NLL={test_loss_log[step]:.4f}"
        )

        # ===========================================
        # Hessian eigenvalues
        # ===========================================
        if eig_freq > 0 and step % eig_freq == 0:
            print("  Computing eigenvalues...")
            eigvals = get_hessian_eigenvalues(
                model, loss_fn, [(train_data, bptt)], neigs=neigs
            )
            eigs[step // eig_freq] = eigvals
            print("  Top eigenvalues:", eigvals.tolist())

        # ===========================================
        # Save intermediate logs
        # ===========================================
        if save_freq > 0 and step % save_freq == 0:
            save_files(
                directory,
                [
                    ("train_loss", train_loss_log[: step + 1]),
                    ("test_loss", test_loss_log[: step + 1]),
                    ("eigs", eigs[: step // eig_freq] if eig_freq > 0 else eigs),
                ],
            )

        # ===========================================
        # Log iterates
        # ===========================================
        if iterate_freq > 0 and step % iterate_freq == 0 and nproj > 0:
            vec = parameters_to_vector(model.parameters()).detach().cpu()
            iterates[step // iterate_freq] = proj_matrix @ vec

        # ===========================================
        # GD update
        # ===========================================
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for i in range(0, train_data.size(0) - 1, bptt):
            data, targets = get_batch(train_data, i, bptt)
            data = data.to(device)
            targets = targets.to(device)

            output = model(data, src_mask)
            output = output.reshape(-1, ntoken)
            targets = targets.reshape(-1)

            batch_loss = loss_fn(output, targets)
            batch_loss.backward()

        optimizer.step()

    # ==========================================================
    # Save final logs
    # ==========================================================
    save_files_final(
        directory,
        [
            ("train_loss", train_loss_log),
            ("test_loss", test_loss_log),
            ("eigs", eigs),
            ("iterates", iterates),
        ],
    )

    if save_model:
        torch.save(model.state_dict(), f"{directory}/snapshot_final")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max_steps", type=int, required=True)

    parser.add_argument("--bptt", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=20)

    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neigs", type=int, default=0)
    parser.add_argument("--eig_freq", type=int, default=-1)
    parser.add_argument("--save_freq", type=int, default=-1)
    parser.add_argument("--nproj", type=int, default=0)
    parser.add_argument("--iterate_freq", type=int, default=-1)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()
    main(**vars(args))
