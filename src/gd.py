import os
import argparse
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from os import makedirs
from hessian import HessianEngine
from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, save_files, save_files_final
from data import load_dataset, take_first, DATASETS


def get_loss_fns(loss_name):
    """
    Returns two loss functions:
    1. For Training/Hessian (Reduction='mean') - standard optimization/analysis
    2. For Accumulation (Reduction='sum') - needed if manually dividing by N later
    """
    if loss_name == "mse":
        return (
            nn.MSELoss(reduction="mean"),
            nn.MSELoss(reduction="sum"),
            lambda x, y: (x.argmax(1) == y.argmax(1)).float().sum(),
        )
    elif loss_name == "ce":
        return (
            nn.CrossEntropyLoss(reduction="mean"),
            nn.CrossEntropyLoss(reduction="sum"),
            lambda x, y: (x.argmax(1) == y).float().sum(),
        )
    raise NotImplementedError(f"Unknown loss: {loss_name}")


def main(
    dataset: str,
    arch_id: str,
    loss: str,
    opt: str,
    lr: float,
    max_steps: int,
    neigs: int = 0,
    physical_batch_size: int = 1000,
    eig_freq: int = -1,
    iterate_freq: int = -1,
    save_freq: int = -1,
    save_model: bool = False,
    beta: float = 0.0,
    nproj: int = 0,
    loss_goal: float = None,
    acc_goal: float = None,
    abridged_size: int = 5000,
    seed: int = 0,
    wd: float = 0,
    resume_model=None,
):

    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, wd, beta)
    print(f"Output directory: {directory}")
    makedirs(directory, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    train_dataset, test_dataset = load_dataset(dataset, loss)

    # If dataset is small (like MNIST/CIFAR), move to GPU once.
    # This removes CPU->GPU transfer latency during the loop.
    if len(train_dataset) < 60000 and hasattr(train_dataset, "tensors"):
        print("Dataset small enough to fit in VRAM. Pre-loading...")
        train_dataset.tensors = tuple(t.to(device) for t in train_dataset.tensors)
        test_dataset.tensors = tuple(t.to(device) for t in test_dataset.tensors)

    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn_mean, loss_fn_sum, acc_fn = get_loss_fns(loss)
    network = load_architecture(arch_id, dataset).to(device)

    if resume_model:
        print(f"Loading pretrained weights: {resume_model}")
        network.load_state_dict(torch.load(resume_model, map_location=device))

    optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta, wd)

    # Init Hessian Engine (Uses 'mean' reduction loss)
    hessian_engine = HessianEngine(
        network, loss_fn_mean, abridged_train, batch_size=physical_batch_size
    )

    # Random Projections for visualization
    # Keep on CPU to save VRAM, or move to GPU if checking freq is high
    projectors = torch.randn(nproj, sum(p.numel() for p in network.parameters()))

    # Pre-allocate tensors
    logs = {
        "train_loss": torch.zeros(max_steps),
        "test_loss": torch.zeros(max_steps),
        "train_acc": torch.zeros(max_steps),
        "test_acc": torch.zeros(max_steps),
        "eigs": torch.zeros(max_steps // eig_freq if eig_freq > 0 else 0, neigs),
        "iterates": torch.zeros(
            max_steps // iterate_freq if iterate_freq > 0 else 0, nproj
        ),
    }

    print(f"Starting training on {device}...")

    # Training Loop
    for step in range(max_steps):

        if eig_freq != -1 and step % eig_freq == 0:
            vals, _ = hessian_engine.compute_dataset_eigenvalues(
                top_k=neigs, max_iter=50, tol=1e-4, return_eigenvectors=False # SET FLAG FOR EIGENVECTORS HERE
            )
            logs["eigs"][step // eig_freq, :] = vals.cpu()
            print(f"Step {step} | Max Eigenvalue: {vals[0]:.4f}")

        if iterate_freq != -1 and step % iterate_freq == 0:
            flat_params = parameters_to_vector(network.parameters()).detach().cpu()
            logs["iterates"][step // iterate_freq, :] = projectors.mv(flat_params)

        with torch.no_grad():
            test_l, test_a = 0.0, 0.0
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=physical_batch_size
            )
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = network(X)
                test_l += loss_fn_sum(preds, y).item()
                test_a += acc_fn(preds, y).item()

            logs["test_loss"][step] = test_l / len(test_dataset)
            logs["test_acc"][step] = test_a / len(test_dataset)

        # Training & Gradient Accumulation Fused
        optimizer.zero_grad()
        train_l, train_a = 0.0, 0.0

        # We use a standard DataLoader for the training loop (handles shuffling if needed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=physical_batch_size, shuffle=True
        )

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            preds = network(X)

            # Loss for optimization
            # Note: We use Sum reduction and divide by N later to mimic Full Batch GD exactly
            # taking 1/N * sum(loss) is equivalent to taking mean of means if batches are equal,
            # but 1/Total * sum(sums) is safer for uneven last batches.
            batch_loss_sum = loss_fn_sum(preds, y)

            # Backward
            # We scale gradients here by 1/Total_Dataset_Size
            (batch_loss_sum / len(train_dataset)).backward()

            # Metrics Accumulation (Avoids extra forward pass)
            train_l += batch_loss_sum.item()
            train_a += acc_fn(preds, y).item()

        logs["train_loss"][step] = train_l / len(train_dataset)
        logs["train_acc"][step] = train_a / len(train_dataset)

        print(
            f"{step}\tTrL: {logs['train_loss'][step]:.4f}\tTrA: {logs['train_acc'][step]:.3f}\t"
            f"TeL: {logs['test_loss'][step]:.4f}\tTeA: {logs['test_acc'][step]:.3f}"
        )

        if save_freq != -1 and step % save_freq == 0:
            current_logs = {
                k: (
                    v[: step + 1]
                    if k in ["train_loss", "test_loss", "train_acc", "test_acc"]
                    else v[: (step // (eig_freq if "eigs" in k else iterate_freq)) + 1]
                )
                for k, v in logs.items()
            }
            save_files(directory, list(current_logs.items()))

        if (loss_goal and logs["train_loss"][step] < loss_goal) or (
            acc_goal and logs["train_acc"][step] > acc_goal
        ):
            print("Goal reached.")
            break

        optimizer.step()

    save_files_final(directory, list(logs.items()))
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Full Batch GD Training")
    parser.add_argument("dataset", type=str, choices=DATASETS)
    parser.add_argument("arch_id", type=str)
    parser.add_argument("loss", type=str, choices=["ce", "mse"])
    parser.add_argument("lr", type=float)
    parser.add_argument("wd", type=float)
    parser.add_argument("max_steps", type=int)
    parser.add_argument(
        "--opt", type=str, default="gd", choices=["gd", "polyak", "nesterov"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--physical_batch_size", type=int, default=1000)
    parser.add_argument("--acc_goal", type=float)
    parser.add_argument("--loss_goal", type=float)
    parser.add_argument("--neigs", type=int, default=0)
    parser.add_argument("--eig_freq", type=int, default=-1)
    parser.add_argument("--nproj", type=int, default=0)
    parser.add_argument("--iterate_freq", type=int, default=-1)
    parser.add_argument("--abridged_size", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=-1)
    parser.add_argument("--save_model", type=bool, default=False)

    args = parser.parse_args()

    main(**vars(args))
