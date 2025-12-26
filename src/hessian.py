import torch
import torch.nn as nn
from torch.func import functional_call, jvp, grad
from torch.nn.utils import parameters_to_vector
import numpy as np
from typing import Tuple, Optional, Union, List


class HessianEngine:
    def __init__(
        self, model: nn.Module, loss_fn: nn.Module, dataset=None, batch_size: int = 128
    ):
        """
        Args:
            model: The model being trained.
            loss_fn: Loss function TODO:(must have reduction='mean').
            dataset: Optional. The full dataset (needed only for compute_dataset_eigenvalues).
            batch_size: Batch size for iterating through the full dataset.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

        # Pre-process params for functional calls
        self.params_dict = dict(model.named_parameters())
        self.param_keys = list(self.params_dict.keys())
        self.param_values = tuple(self.params_dict.values())
        self.num_params = sum(p.numel() for p in self.param_values)

    def compute_dataset_eigenvalues(
        self,
        top_k: int = 2,
        max_iter: int = 100,
        tol: float = 1e-5,
        return_eigenvectors: bool = False,
    ):
        """
        Calculates sharpness on the ENTIRE dataset (Global Landscape).
        Iterates through self.dataset using a DataLoader.
        """
        if self.dataset is None:
            raise ValueError("HessianEngine was initialized without a dataset.")

        def mv_closure(v):
            return self._compute_hvp_dataset(v)

        return self._run_lanczos(mv_closure, top_k, max_iter, tol, return_eigenvectors)

    def compute_batch_eigenvalues(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        top_k: int = 2,
        max_iter: int = 100,
        tol: float = 1e-5,
        return_eigenvectors: bool = False,
    ):
        """
        Calculates sharpness on a SPECIFIC BATCH (Local Landscape).
        Useful for:
          1. Checking the sharpness of the actual step taken (Batch Sharpness).
          2. Full Batch GD (batch_data is full dataset) equivalent to full dataset method.
        """
        X, y = batch_data
        X, y = X.to(self.device), y.to(self.device)

        def mv_closure(v):
            return self._compute_hvp_single_batch(v, X, y)

        return self._run_lanczos(mv_closure, top_k, max_iter, tol, return_eigenvectors)

    def compute_spectral_density(
        self,
        batch_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_probes: int = 10,
        max_iter: int = 50,
        sigma: float = 0.01,
    ):
        """
        Estimates full spectrum density (ESD).
        If batch_data is None, uses full dataset.
        """
        all_evals = []
        all_weights = []

        def mv_closure(v):
            if batch_data is not None:
                X, y = batch_data
                return self._compute_hvp_single_batch(
                    v, X.to(self.device), y.to(self.device)
                )
            else:
                return self._compute_hvp_dataset(v)

        for _ in range(n_probes):
            # Always use fast Lanczos (no eigenvectors needed)
            alpha, beta = self._lanczos_fast(mv_closure, max_iter, tol=1e-4)

            # Solve small system on CPU
            T = torch.diag(alpha) + torch.diag(beta[:-1], 1) + torch.diag(beta[:-1], -1)
            eigvals, eigvecs = torch.linalg.eigh(T)

            # SLQ weights
            weights = eigvecs[0, :] ** 2

            all_evals.append(eigvals.cpu().numpy())
            all_weights.append(weights.cpu().numpy())

        return self._smear_density(all_evals, all_weights, sigma)

    # =========================================================================

    def _compute_hvp_single_batch(self, vector, X, y):
        """HVP on a single tensor pair (Batch or Full Batch loaded in memory)"""
        tangents = self._reshape_flat(vector)

        def f_loss(params_tuple, x_batch, y_batch):
            p_dict = {k: v for k, v in zip(self.param_keys, params_tuple)}
            out = functional_call(self.model, p_dict, (x_batch,))
            return self.loss_fn(out, y_batch)

        g_loss_fn = grad(f_loss)

        def gradient_closure(p):
            return g_loss_fn(p, X, y)

        with torch.autocast(device_type=self.device.type):
            _, batch_hvp_tuple = jvp(
                gradient_closure, (self.param_values,), (tangents,)
            )

        return parameters_to_vector(batch_hvp_tuple)

    def _compute_hvp_dataset(self, vector):
        """HVP iterating over the dataset loader"""
        tangents = self._reshape_flat(vector)
        total_hvp = torch.zeros_like(vector)
        total_count = 0

        # Helper for JVP
        def f_loss(params_tuple, x_batch, y_batch):
            p_dict = {k: v for k, v in zip(self.param_keys, params_tuple)}
            out = functional_call(self.model, p_dict, (x_batch,))
            return self.loss_fn(out, y_batch)

        g_loss_fn = grad(f_loss)

        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            N = X.shape[0] if X.dim() > 2 else X.shape[1]

            def gradient_closure(p):
                return g_loss_fn(p, X, y)

            with torch.autocast(device_type=self.device.type):
                _, batch_hvp_tuple = jvp(
                    gradient_closure, (self.param_values,), (tangents,)
                )

            batch_hvp_flat = parameters_to_vector(batch_hvp_tuple)

            # Weighted accumulation
            total_hvp += batch_hvp_flat.float() * N
            total_count += N

        return total_hvp / total_count

    # Lanczos Implementations
    
    def _run_lanczos(self, mv_func, k, max_iter, tol, return_eigenvectors):
        """Dispatches to the correct Lanczos implementation based on requirements"""

        # OPTIMIZATION: If we don't need eigenvectors, do NOT store the basis.
        # This saves VRAM: O(k * N) instead of O(max_iter * N)
        if not return_eigenvectors:
            alpha, beta = self._lanczos_fast(mv_func, max_iter, tol)

            # Solve T matrix for eigenvalues only
            T = torch.diag(alpha) + torch.diag(beta[:-1], 1) + torch.diag(beta[:-1], -1)
            eigvals = torch.linalg.eigvalsh(T)  # faster than eigh

            # Sort descending
            idx = torch.argsort(eigvals, descending=True)[:k]
            return eigvals[idx], None  # No eigenvectors

        else:
            # We need eigenvectors, so we must use the storage-heavy version
            return self._lanczos_exact_with_vectors(mv_func, k, max_iter, tol)

    def _lanczos_fast(self, mv_func, max_iter, tol):
        """Memory Efficient: Discards basis vectors immediately."""
        alpha = torch.zeros(max_iter, device=self.device)
        beta = torch.zeros(max_iter, device=self.device)

        v_curr = torch.randn(self.num_params, device=self.device)
        v_curr /= torch.norm(v_curr)
        v_prev = torch.zeros_like(v_curr)

        for i in range(max_iter):
            w = mv_func(v_curr)
            al = torch.dot(v_curr, w)
            alpha[i] = al

            w = w - al * v_curr - (beta[i - 1] if i > 0 else 0) * v_prev
            bt = torch.norm(w)

            if bt < tol:
                # Truncate alpha/beta to actual size and return
                return alpha[: i + 1], beta[:i]

            if i < max_iter - 1:
                beta[i] = bt

            v_prev = v_curr.clone()
            v_curr = w / bt

        return alpha, beta

    def _lanczos_exact_with_vectors(self, mv_func, k, max_iter, tol):
        """Storage Heavy: Keeps Q matrix to reconstruct eigenvectors."""
        Q = torch.zeros((max_iter, self.num_params), device=self.device)
        alpha = torch.zeros(max_iter, device=self.device)
        beta = torch.zeros(max_iter, device=self.device)

        v = torch.randn(self.num_params, device=self.device)
        v = v / torch.norm(v)
        Q[0] = v

        iters_run = 0
        for i in range(max_iter):
            iters_run = i + 1
            w = mv_func(Q[i])
            alpha[i] = torch.dot(Q[i], w)

            w = w - alpha[i] * Q[i]
            if i > 0:
                w = w - beta[i - 1] * Q[i - 1]

            bt = torch.norm(w)
            if bt < tol:
                break

            if i < max_iter - 1:
                beta[i] = bt
                Q[i + 1] = w / bt

        # Reconstruct
        T = (
            torch.diag(alpha[:iters_run])
            + torch.diag(beta[: iters_run - 1], 1)
            + torch.diag(beta[: iters_run - 1], -1)
        )
        eigvals, eigvecs = torch.linalg.eigh(T)

        idx = torch.argsort(eigvals, descending=True)[: min(k, iters_run)]
        top_evals = eigvals[idx]

        # Map back: V = Q @ v_tridiag
        top_evecs = torch.matmul(Q[:iters_run].T, eigvecs[:, idx])

        return top_evals, top_evecs

    # HELPERS
    
    def _reshape_flat(self, flat_vec):
        pointer = 0
        tuple_views = []
        for p in self.param_values:
            numel = p.numel()
            tuple_views.append(flat_vec[pointer : pointer + numel].view_as(p))
            pointer += numel
        return tuple(tuple_views)

    def _smear_density(self, evals_list, weights_list, sigma):
        all_evals = np.concatenate(evals_list)
        all_weights = np.concatenate(weights_list)
        min_e, max_e = all_evals.min(), all_evals.max()
        margin = max(abs(min_e), abs(max_e)) * 0.1
        grid = np.linspace(min_e - margin, max_e + margin, 1000)
        density = np.zeros_like(grid)
        for mu, w in zip(all_evals, all_weights):
            density += w * np.exp(-((grid - mu) ** 2) / (2 * sigma**2))
        density /= np.sqrt(2 * np.pi) * sigma * len(evals_list)
        return grid, density
