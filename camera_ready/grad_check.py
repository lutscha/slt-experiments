"""Check grad S from compute_cy against finite differences of S, for the MLPs of the layernorm group.

Usage: python camera_ready/grad_check.py; exits non-zero on failure.
Two directions are checked: theta itself (<grad S, theta> = c_y), and a joint scaling of the layers that feed into
a LayerNorm (for full_lnh_200 this is exactly -2 S q by scale invariance).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
from torch.nn.utils import parameters_to_vector, vector_to_parameters  # noqa: E402
from torch.utils.data import TensorDataset  # noqa: E402

from archs import load_architecture  # noqa: E402
from utilities import get_hessian_eigenvalues, compute_cy, SquaredLoss  # noqa: E402

INVARIANT_PREFIXES = ("1.", "4.")
EPS = 1e-2
TOL = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
n = 256
data = TensorDataset(torch.randn(n, 3, 32, 32, device=device),
                     torch.eye(10, device=device)[torch.randint(0, 10, (n,), device=device)])
loss_fn = SquaredLoss()
failures = []

for arch_id in ("full_200", "full_lnh_200"):
    torch.manual_seed(1)
    net = load_architecture(arch_id, "cifar10").to(device)
    theta = parameters_to_vector(net.parameters()).detach().clone()
    inv = torch.cat([torch.full((p.numel(),), float(nm.startswith(INVARIANT_PREFIXES)), device=device)
                     for nm, p in net.named_parameters()])

    def S_at(t):
        vector_to_parameters(t, net.parameters())
        return get_hessian_eigenvalues(net, loss_fn, data, neigs=1)[0][0].item()

    evals, evecs = get_hessian_eigenvalues(net, loss_fn, data, neigs=1)
    S = evals[0].item()
    c_y, _, (cy_b, u2_b, _) = compute_cy(net, loss_fn, data, evecs, per_block=True)
    mask = torch.tensor([nm.startswith(INVARIANT_PREFIXES) for nm, _ in net.named_parameters()])
    checks = [("theta", theta, c_y), ("invariant blocks", theta * inv, cy_b[mask].sum().item())]
    for name, v, autograd in checks:
        fd = (S_at(theta + EPS * v) - S_at(theta - EPS * v)) / (2 * EPS)
        ok = abs(autograd - fd) <= TOL * max(abs(fd), 0.1 * S)
        print(f"{'ok  ' if ok else 'FAIL'}  {arch_id:12s} direction={name:16s} autograd={autograd:10.4f}  "
              f"finite diff={fd:10.4f}  (S={S:.4f}, -2Sq={-2 * S * u2_b[mask].sum().item():.4f})")
        if not ok:
            failures.append(f"{arch_id} {name}")
    vector_to_parameters(theta, net.parameters())

print("GRAD CHECK FAILED: " + ", ".join(failures) if failures else "GRAD CHECK PASSED")
raise SystemExit(1 if failures else 0)
