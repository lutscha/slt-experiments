from typing import List, Tuple, Iterable

import math
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import os
from transformer_lm.lm_model import generate_square_subsequent_mask
import time
# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000


def get_gd_directory(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str,wd: float, beta: float = None):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/{opt}"
    if opt in ("gd", "sgd"):
        return f"{directory}/lr_{lr}/wd_{wd}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{directory}/lr_{lr}_beta_{beta}"


def get_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/flow/tick_{tick}"


def get_modified_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, gd_lr: float, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/modified_flow_lr_{gd_lr}/tick_{tick}"


def get_gd_optimizer(parameters, opt: str, lr: float, momentum: float, wd: float) -> Optimizer:
    if opt == "gd":
        return SGD(parameters, lr=lr, weight_decay=wd)
    elif opt == "polyak":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=False)
    elif opt == "nesterov":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=True)


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}.pt")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_final.pt")


# def iterate_dataset(dataset: Dataset, batch_size: int):
#     """Iterate through a dataset, yielding batches of data."""
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     for (batch_X, batch_y) in loader:
#         yield batch_X.to(device), batch_y.to(device)

def iterate_dataset(dataset, batch_size):
    """
    Iterate through dataset, yielding batches of data.
    Detects when dataset samples are already batched (LM case)
    and skips DataLoader batching in that case.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Peek at the first sample
    sample_X, sample_y = dataset[0]

    # --- CASE 1: LM dataset (samples are already mini-batches) ---
    if sample_X.dim() == 2:     # X is (seq_len, batch)
        for X, y in dataset:
            yield X.to(device), y.to(device)
        return

    # --- CASE 2: regular dataset (MLP/CNN), use normal mini-batching ---
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_X, batch_y in loader:
        yield batch_X.to(device), batch_y.to(device)

def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses


def get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()

    raise NotImplementedError(f"no such loss function: {loss}")
    


# def compute_hvp(network: nn.Module, loss_fn: nn.Module,
#                 dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS, P: Tensor = None):
#     """Compute a Hessian-vector product.
    
#     If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     p = len(parameters_to_vector(network.parameters()))
#     n = len(dataset)
#     hvp = torch.zeros(p, dtype=torch.float, device=device)
#     vector = vector.to(device)
#     if P is not None:
#         vector = vector / P.to(device).sqrt()
#     for (X, y) in iterate_dataset(dataset, physical_batch_size):
#         loss = loss_fn(network(X), y) / n
#         grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
#         dot = parameters_to_vector(grads).mul(vector).sum()
#         grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
#         hvp += parameters_to_vector(grads)
#     if P is not None:
#         hvp = hvp / P.to(device).sqrt()
#     return hvp

def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS, P: Tensor = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    if P is not None:
        vector = vector / P.to(device).sqrt()

    # Detect if this is a language model dataset (already mini-batches)
    sample_X, sample_y = dataset[0]
    is_lm_dataset = sample_X.dim() == 2

    if is_lm_dataset:
        # ============================================================
        # LANGUAGE MODEL CASE (seq_len x batch_size)
        # ============================================================
        for X, y in dataset:
            X = X.to(device)
            y = y.to(device)

            seq_len = X.size(0)
            src_mask = generate_square_subsequent_mask(seq_len).to(device)

            logits = network(X, src_mask)                # (seq, batch, vocab)
            logits = logits.reshape(-1, logits.size(-1)) # (seq*batch, vocab)
            y_flat = y.reshape(-1)                       # (seq*batch)

            loss = loss_fn(logits, y_flat) / n

            grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
            dot = parameters_to_vector(grads).mul(vector).sum()
            grads = [g.contiguous()
                     for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
            hvp += parameters_to_vector(grads)

    else:
        # ============================================================
        # CNN / MLP CASE (normal samples, use batching)
        # ============================================================
        for X, y in iterate_dataset(dataset, physical_batch_size):
            logits = network(X)
            loss = loss_fn(logits, y) / n

            grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
            dot = parameters_to_vector(grads).mul(vector).sum()
            grads = [g.contiguous()
                     for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
            hvp += parameters_to_vector(grads)

    if P is not None:
        hvp = hvp / P.to(device).sqrt()

    return hvp


def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, physical_batch_size=1000, P=None):
    """ Compute the leading Hessian eigenvalues.
    
    If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size, P=P).detach().cpu()
    
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)

    # trace_est = trace_estimate(hvp_delta, nparams) #Estimate trace via Hutchinson estimator
    # low_trace = trace_est - evals.sum().item() #Estimates sum of N-50 eigenvalues of Hessian

    return evals, evecs

def flatt(vectors):
    '''
    Flattens a list of vectors into a single vector
    '''
    return torch.cat([v.flatten() for v in vectors])


def compute_rayleigh_quotient(model, loss):
    """Compute g^T H g / g^T g for one batch (inputs, targets) at the current model params."""
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    grads_vector = flatt(grads)
    step_vector = grads_vector.detach()

    grad_step = torch.dot(grads_vector, step_vector)

    # Compute Hessian-vector product H*g (by taking gradient of the dot(grad, grad_outputs))
    Hv = torch.autograd.grad(grad_step, model.parameters(), retain_graph=False)
    Hv = flatt(Hv).detach()
    
    return (torch.dot(step_vector, Hv), torch.dot(step_vector, step_vector))


def estimate_batch_sharpness(model, 
                             X,
                             Y, 
                             loss_fn, 
                             batch_size, 
                             max_estimates=500, 
                             eps=0.005):
    """Estimate E[g^T H g / g^T g] over random batches via Monte Carlo sampling."""
    model.eval()  # Ensure model is in eval mode (no dropout, etc.) for consistency
    device = next(model.parameters()).device
    gHg_vals = []
    norm_g_vals = []

    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)

    for i in range(max_estimates):

        shuffle = torch.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx].to(device)
        Y_batch = Y[random_idx].to(device)

        loss = loss_fn(model(X_batch), Y_batch)/Y_batch.size(0)

        gHg, norm_g = compute_rayleigh_quotient(model, loss)

        gHg = gHg.item()
        norm_g = norm_g.item()

        gHg_vals.append(gHg)
        norm_g_vals.append(norm_g)

        if i < 20:
            continue
        
        mean_x, mean_y = np.mean(gHg_vals), np.mean(norm_g_vals)
        var_x,  var_y  = np.var(gHg_vals, ddof=1), np.var(norm_g_vals, ddof=1)
        cov_xy = np.cov(gHg_vals, norm_g_vals, ddof=1)[0, 1]
        
        R = mean_x / mean_y

        var_R = (var_x / mean_y**2
                 - 2 * cov_xy * mean_x / mean_y**3
                 + var_y * mean_x**2 / mean_y**4) / len(gHg_vals)
        
        rse = np.sqrt(var_R) / abs(R)  # relative standard error

        if rse < eps:                    # stopping rule
            break

    gHg_normalized = np.array(gHg_vals) / np.array(norm_g_vals)
    return float(np.mean(gHg_normalized))
    
# def compute_rayleigh_quotient(model, loss_fn, inputs, targets):
#     """Compute g^T H g / g^T g for one batch (inputs, targets) at the current model params."""
#     model.zero_grad(set_to_none=True)
#     # Forward pass and compute loss (assumed mean loss over the batch)
    
#     outputs = model(inputs)

#     B=targets.size(0)
#     loss = loss_fn(outputs, targets) / B #Use "sum" loss, so divide by B to make it mean
#     # Compute gradients w.r.t. parameters (creating graph for second-order calc)
#     grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

#     v = [g.detach() for g in grad_params]
#     # Compute Hessian-vector product H*g (by taking gradient of the dot(grad, grad_outputs))
#     hv = torch.autograd.grad(grad_params, model.parameters(), grad_outputs=v, retain_graph=False)
#     # Compute g^T H g (dot product of grad and H*grad), and grad norm squared
#     gHg = sum((vi * hvi).sum() for vi, hvi in zip(v, hv))
#     grad_norm_sq = sum((vi * vi).sum() for vi in v)
#     # Rayleigh quotient (add a tiny epsilon for safety to avoid division by zero)
#     # return (gHg / (grad_norm_sq + 1e-12)).item()
#     return (gHg.item(), grad_norm_sq.item())

# def estimate_batch_sharpness(model, data_loader, loss_fn, max_batches=500, rel_error_tol=0.005):
#     """Estimate E[g^T H g / g^T g] over random batches via Monte Carlo sampling."""
#     model.eval()  # Ensure model is in eval mode (no dropout, etc.) for consistency
#     sum_gHg, sum_g2, count = 0.0, 0.0, 0
#     sum_rq, sum_sq = 0.0, 0.0
#     for inputs, targets in data_loader:
#         inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
#         # Compute Rayleigh quotient for this batch
#         # rq = compute_rayleigh_quotient(model, loss_fn, inputs, targets)
#         gHg, g2 = compute_rayleigh_quotient(model, loss_fn, inputs, targets)
#         # Update running average and variance
#         count += 1
        
#         sum_gHg += gHg
#         sum_g2 += g2

#         rq = gHg / g2 + 1e-12
#         sum_rq += rq
#         sum_sq += rq * rq

#         if count >= 2:  # check relative error after at least 2 samples
#             mean = sum_rq / count
#             # (Using standard error of the mean as uncertainty estimate)
#             variance = (sum_sq / count) - mean**2
#             std_error = math.sqrt(abs(variance) / count)
#             if std_error / abs(mean) < rel_error_tol:
#                 break
#         if count >= max_batches:
#             break
#     return sum_gHg / sum_g2










def trace_estimate(hvp_fun, nparams, num_samples=30):
    trace_est=0.0
    for _ in range(num_samples):
        z = torch.randn(nparams)
        Hz = hvp_fun(z)
        trace_est +=torch.dot(z, Hz)
    return trace_est.item() / num_samples


def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at the current network parameters. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device=device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient


class AtParams(object):
    """ Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def compute_gradient_at_theta(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              theta: torch.Tensor, batch_size=DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at arbitrary network parameters "theta".  """
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, physical_batch_size=batch_size)


class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()


class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()


class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()


class VoidLoss(nn.Module):
    def forward(self, X, Y):
        return 0



def make_batch_stepper(dataset, batch_size, shuffle=True):
    """
    Returns a function next_batch() that yields ONE batch each time.
    For normal datasets uses DataLoader; for LM datasets (already batched) cycles dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_X, sample_y = dataset[0]

    # LM case: dataset items are already batches
    if sample_X.dim() == 2:
        i = 0
        n = len(dataset)

        def next_batch():
            nonlocal i
            X, y = dataset[i]
            i = (i + 1) % n
            return X.to(device), y.to(device)

        return next_batch

    # Regular case: DataLoader batching
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(loader)  # new epoch (reshuffles if shuffle=True)
            X, y = next(it)
        return X.to(device), y.to(device)

    return next_batch









