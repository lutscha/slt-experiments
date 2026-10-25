import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev
from torch.utils.data import TensorDataset


def _jacobian(network: nn.Module, params: dict, X: torch.Tensor) -> torch.Tensor:
    """Jacobian of the network outputs w.r.t. all parameters, rows ordered (sample, class): (B*C, P)."""
    def f(p, x):
        return functional_call(network, p, (x.unsqueeze(0),)).squeeze(0)

    jac = vmap(jacrev(f), in_dims=(None, 0))(params, X)
    B = X.shape[0]
    return torch.cat([jac[name].reshape(B * jac[name].shape[1], -1) for name in params], dim=1)


def empirical_ntk(network: nn.Module, X: torch.Tensor, chunk: int = 50) -> torch.Tensor:
    """Empirical NTK K = J J^T / n of size (n*C, n*C), with J the (n*C, P) Jacobian on n samples.

    This is the Gauss-Newton convention for the mean loss, so lambda_max(K) is comparable to the sharpness.
    K is built block by block, so the full Jacobian is never held in memory.
    """
    device = next(network.parameters()).device
    params = {name: p.detach() for name, p in network.named_parameters()}
    X = X.to(device)
    n = X.shape[0]
    starts = list(range(0, n, chunk))
    C = network(X[:1]).shape[1]
    K = torch.zeros(n * C, n * C, device=device)
    with torch.no_grad():
        for ia, a in enumerate(starts):
            Ja = _jacobian(network, params, X[a:a + chunk])
            ra = slice(a * C, a * C + Ja.shape[0])
            for b in starts[ia:]:
                Jb = Ja if b == a else _jacobian(network, params, X[b:b + chunk])
                rb = slice(b * C, b * C + Jb.shape[0])
                block = Ja @ Jb.T
                K[ra, rb] = block
                K[rb, ra] = block.T
                del Jb
            del Ja
    return K / n


def _alignment(K: torch.Tensor, y: torch.Tensor) -> float:
    """Kernel target alignment <K, y y^T>_F / (||K||_F ||y y^T||_F)."""
    return ((y @ K @ y) / (torch.linalg.norm(K) * (y @ y))).item()


def ntk_metrics(network: nn.Module, dataset: TensorDataset, chunk: int = 50) -> dict:
    """Top eigenvalue of the empirical NTK and kernel target alignment (KTA) on the given samples.

    Reported alignments, with Y the (n, C) one-hot MSE targets:
      kta          : full (n*C)x(n*C) NTK against vec(Y)
      kta_centered : same, with class-mean-centered targets
      kta_trace    : class-summed NTK K_ij = sum_c K_(i,c),(j,c) against Y Y^T
    """
    X, Y = dataset.tensors
    K = empirical_ntk(network, X, chunk=chunk)
    n, C = Y.shape
    y = Y.to(K.device).float()
    y_c = y - y.mean(0, keepdim=True)
    K_trace = K.reshape(n, C, n, C).diagonal(dim1=1, dim2=3).sum(-1)
    YY = y @ y.T
    return {
        "lambda_max": torch.linalg.eigvalsh(K)[-1].item(),
        "kta": _alignment(K, y.reshape(-1)),
        "kta_centered": _alignment(K, y_c.reshape(-1)),
        "kta_trace": ((K_trace * YY).sum() / (torch.linalg.norm(K_trace) * torch.linalg.norm(YY))).item(),
        "n": n,
    }
