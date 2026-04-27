"""Analytical 3D toy loss L(x, y, z) and GD dynamics with tracked sharpness quantities.

In L, η is the GD learning rate: A = 2/η + √β y, so L, ∇L, S, and ∇S all depend on the
same η used in the update xyz ← xyz − η ∇L.
"""

import numpy as np
from scipy.integrate import solve_ivp

def _intermediates(
    x: float | np.ndarray,
    y: float | np.ndarray,
    eta: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sqrt_beta = np.sqrt(beta)
    A = 2.0 / eta + sqrt_beta * y
    B = sqrt_beta * x
    R = np.sqrt(A * A + 4.0 * B * B)
    return A, B, R


def L(x, y, z, eta: float, alpha: float, beta: float) -> np.ndarray:
    """L = ((2/η) + sqrt(beta)*y) * x^2/2 - (alpha/sqrt(beta))*y - z; η is the GD learning rate."""
    A, _, _ = _intermediates(x, y, eta, beta)
    sqrt_beta = np.sqrt(beta)
    return A * (x * x) * 0.5 - (alpha / sqrt_beta) * y - z


def grad_L(x, y, z, eta: float, alpha: float, beta: float) -> np.ndarray:
    A, _, _ = _intermediates(x, y, eta, beta)
    sqrt_beta = np.sqrt(beta)
    gx = A * x
    gy = sqrt_beta * 0.5 * (x * x) - alpha / sqrt_beta
    gz = np.full_like(x, -1.0, dtype=float)
    return np.stack((gx, gy, gz), axis=-1)


def S(x, y, z, eta: float, alpha: float, beta: float) -> np.ndarray:
    """Sharpness = largest Hessian eigenvalue: (A + R) / 2."""
    A, _, R = _intermediates(x, y, eta, beta)
    return 0.5 * (A + R)


def grad_S(x, y, z, eta: float, alpha: float, beta: float, eps: float = 1e-30) -> np.ndarray:
    A, _, R = _intermediates(x, y, eta, beta)
    sqrt_beta = np.sqrt(beta)
    inv_R = 1.0 / np.maximum(R, eps)
    gx = 2.0 * beta * x * inv_R
    gy = sqrt_beta * 0.5 * (1.0 + A * inv_R)
    gz = np.zeros_like(x, dtype=float)
    return np.stack((gx, gy, gz), axis=-1)


def u(x, y, z, eta: float, alpha: float, beta: float) -> np.ndarray:
    s = S(x, y, z, eta, alpha, beta)
    B = np.sqrt(beta) * x
    return np.stack((s, B, np.zeros_like(x, dtype=float)), axis=-1)


def u_max(
    x: float,
    y: float,
    z: float,
    eta: float,
    alpha: float,
    beta: float,
    eps: float = 1e-30,
) -> np.ndarray:
    """Unit eigenvector for λ_max(H) in the (x, y) block: parallel to (S, √β x, 0)."""
    u_vec = np.asarray(u(x, y, z, eta, alpha, beta), dtype=float).reshape(3)
    nrm = float(np.linalg.norm(u_vec))
    if nrm < eps:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return u_vec / nrm


def gd_run(
    x0: float,
    y0: float,
    z0: float,
    lr: float,
    wd: float,
    alpha: float,
    beta: float,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """
    Full-batch GD with weight decay: θ ← θ − η∇L − ηλ(θ − θ_off), η = lr in L, ∇L, S, u.

    Live diagnostics (no θ*): u from u_max, α = −∇L·∇S, β_S = ‖∇S‖²,
    """
    eta = lr
    xyz = np.array([x0, y0, z0], dtype=float)
    n = n_steps + 1
    history = {
        "L": np.empty(n, dtype=float),
        "grad_L": np.empty((n, 3), dtype=float),
        "S": np.empty(n, dtype=float),
        "grad_S": np.empty((n, 3), dtype=float),
        "u": np.empty((n, 3), dtype=float),
        "xyz": np.empty((n, 3), dtype=float),
        "alpha": np.empty(n, dtype=float),
        "beta_S": np.empty(n, dtype=float),
        "c_x_live": np.empty(n, dtype=float),
        "c_y_live": np.empty(n, dtype=float),
        "abs_c_x_live": np.empty(n, dtype=float),
    }
    for t in range(n):
        x, y, z = xyz
        history["L"][t] = float(L(x, y, z, eta, alpha, beta))
        gL = grad_L(x, y, z, eta, alpha, beta)
        gS = grad_S(x, y, z, eta, alpha, beta)
        uu = u_max(x, y, z, eta, alpha, beta)
        history["grad_L"][t] = gL
        history["S"][t] = float(S(x, y, z, eta, alpha, beta))
        history["grad_S"][t] = gS
        history["u"][t] = uu
        history["xyz"][t] = xyz.copy()
        history["alpha"][t] = -float(np.dot(gL, gS))
        history["beta_S"][t] = float(np.dot(gS, gS))
        history["c_x_live"][t] = float(np.dot(uu, xyz))
        history["c_y_live"][t] = float(np.dot(xyz, gS))
        history["abs_c_x_live"][t] = abs(history["c_x_live"][t])
        if t < n_steps:
            xyz = xyz - lr * gL - lr * wd * xyz
    return history

def predict_dynamics(
    x0: float,
    y0: float,
    lr: float,
    wd: float,
    alpha: float,
    beta: float,
    n_steps: int,
    c_x: float = 0.0,
    c_y: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """
    Computes the theoretical continuous ODE dynamics for the Edge of Stability 
    with Weight Decay, evaluated at discrete steps t.
    """
    # 1. Define the Continuous ODE system
    def eos_ode(tau, state):
        X, Y = state
        dX = X * (Y + wd)
        dY = alpha - 0.5 * beta * (X**2) - wd * Y - wd * c_y
        return [dX, dY]

    # 2. Setup initial conditions for the macroscopic envelope
    # Local basis mapping
    Y0 = np.sqrt(beta) * y0
    
    # The initial amplitude envelope X(0) absorbs the c_x shift
    X0 = abs(x0 + (lr * wd * c_x) / 2.0)
    
    # 3. Set up the time evaluation points (tau = eta * t)
    taus = np.arange(n_steps + 1) * lr
    tau_max = taus[-1]

    # 4. Integrate the continuous ODE
    sol = solve_ivp(
        fun=eos_ode,
        t_span=(0.0, tau_max),
        y0=[X0, Y0],
        t_eval=taus,
        method='RK45', # Runge-Kutta 4th order
        rtol=1e-8,
        atol=1e-10
    )
    
    X_tau = sol.y[0]
    Y_tau = sol.y[1]
    
    # 5. Translate the continuous envelope back to the predicted trajectory
    signs = np.array([(-1)**t for t in range(n_steps + 1)])
    
    preds = {}
    # x_t = (-1)^t * X(tau) - (eta * lambda * c_x) / 2
    preds["x_pred"] = signs * X_tau - (lr * wd * c_x) / 2.0
    
    # S_t = 2/eta + Y(tau)
    preds["S_pred"] = 2.0 / lr + Y_tau
    
    # Continuous Envelopes (for plotting the upper/lower bounds cleanly)
    preds["X_env_upper"] = X_tau - (lr * wd * c_x) / 2.0
    preds["X_env_lower"] = -X_tau - (lr * wd * c_x) / 2.0
    
    # Theoretical steady state values
    preds["S_resting"] =2.0 / lr - wd
    val = 2.0 * (alpha + wd**2 - wd * c_y) / beta
    preds["X_resting"] = np.sqrt(val) if val > 0 else 0.0
    
    return preds