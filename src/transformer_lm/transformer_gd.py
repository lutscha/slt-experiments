import os
import torch
import argparse
from torch.nn.utils import parameters_to_vector

from ..utilities import (
    get_gd_directory,
    get_gd_optimizer,
    get_hessian_eigenvalues,
    save_files,
    save_files_final,
)

from .lm_model import TransformerLM, generate_square_subsequent_mask
from .lm_dataset import load_wikitext2_lm


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

    arch_id = "transformer"
    loss_type = "nll"

    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss_type, wd)
    os.makedirs(directory, exist_ok=True)
    print(f"\nSaving results to:\n  {directory}\n")

    torch.manual_seed(seed)

    # Load HuggingFace WikiText2 dataset
    train_iter, valid_iter, test_iter, vocab = load_wikitext2_lm(
    bptt=bptt, batch_size=batch_size
    ) 
    train_dataset = train_iter
    valid_dataset = valid_iter
    test_dataset  = test_iter

    ntoken = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Vocab size = {ntoken}")
    print(f"Train tokens = {len(train_dataset) * bptt}")
    print(f"Test  tokens = {len(test_dataset) * bptt}")

    model = TransformerLM(
        ntoken=ntoken,
        ninp=200,
        nhead=2,
        nhid=200,
        nlayers=2,
        dropout=0.0,
    ).to(device)

    loss_fn = torch.nn.NLLLoss(reduction="sum")
    optimizer = get_gd_optimizer(model.parameters(), opt, lr, momentum=0.0, wd=wd)

    train_loss_log = torch.zeros(max_steps)
    test_loss_log = torch.zeros(max_steps)
    eigs = torch.zeros(max_steps // eig_freq if eig_freq > 0 else 0, neigs)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, nproj)

    if nproj > 0:
        proj_matrix = torch.randn(nproj, len(parameters_to_vector(model.parameters())))

    # ================================================================
    # TRAINING LOOP
    # ================================================================
    for step in range(max_steps):

        # -----------------------------------
        # Compute TRAIN LOSS (full-batch)
        # -----------------------------------
        train_loss = 0.0
        with torch.no_grad():
            for X, Y in train_dataset:
                X = X.to(device)
                Y = Y.to(device)
                  
                seq_len = X.size(0)
                src_mask = generate_square_subsequent_mask(seq_len).to(device)

                out = model(X, src_mask)
                out = out.reshape(-1, ntoken)
                y_flat = Y.reshape(-1)

                train_loss += loss_fn(out, y_flat)

        train_loss_log[step] = train_loss / len(train_dataset)

        # -----------------------------------
        # Compute TEST LOSS
        # -----------------------------------
        test_loss = 0.0
        with torch.no_grad():
            for X, Y in test_dataset:
                X = X.to(device)
                Y = Y.to(device)

                seq_len = X.size(0)
                src_mask = generate_square_subsequent_mask(seq_len).to(device)

                out = model(X, src_mask)
                out = out.reshape(-1, ntoken)
                y_flat = Y.reshape(-1)

                test_loss += loss_fn(out, y_flat)

        test_loss_log[step] = test_loss / len(test_dataset)

        print(
            f"{step}\t"
            f"Train NLL={train_loss_log[step]:.4f}\t"
            f"Test NLL={test_loss_log[step]:.4f}"
        )

        # -----------------------------------
        # HESSIAN EIGENVALUES
        # -----------------------------------
        if eig_freq > 0 and step % eig_freq == 0:
            print("  Computing eigenvalues...")
            eigvals = get_hessian_eigenvalues(
                model, loss_fn, train_dataset, neigs=neigs
            )
            eigs[step // eig_freq] = eigvals
            print("  Top eigenvalues:", eigvals.tolist())

        # -----------------------------------
        # SAVE INTERMEDIATE LOGS
        # -----------------------------------
        if save_freq > 0 and step % save_freq == 0:
            save_files(
                directory,
                [
                    ("train_loss", train_loss_log[: step + 1]),
                    ("test_loss", test_loss_log[: step + 1]),
                    ("eigs", eigs[: step // eig_freq] if eig_freq > 0 else eigs),
                ],
            )

        # -----------------------------------
        # ITERATE LOGGING
        # -----------------------------------
        if iterate_freq > 0 and step % iterate_freq == 0 and nproj > 0:
            iter_vec = parameters_to_vector(model.parameters()).detach().cpu()
            iterates[step // iterate_freq] = proj_matrix @ iter_vec

        # ================================================================
        # FULL-BATCH GRADIENT DESCENT UPDATE
        # ================================================================
        optimizer.zero_grad(set_to_none=True)

        for X, Y in train_dataset:
            X = X.to(device)
            Y = Y.to(device)

            seq_len = X.size(0)
            src_mask = generate_square_subsequent_mask(seq_len).to(device)

            out = model(X, src_mask)
            out = out.reshape(-1, ntoken)
            y_flat = Y.reshape(-1)

            batch_loss = loss_fn(out, y_flat)
            batch_loss.backward()

        optimizer.step()

    # ================================================================
    # SAVE FINAL RESULTS
    # ================================================================
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

