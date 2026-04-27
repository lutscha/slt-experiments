import torch
import torch.nn.utils as nn_utils
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.func import functional_call

# ==========================================
# --- 1. Helpers ---
# ==========================================

def dict_to_vector(param_dict):
    """Flattens a dictionary of tensors into a single 1D vector."""
    return torch.cat([v.flatten() for v in param_dict.values()])

def vector_to_dict(vec, template_dict):
    """Reshapes a 1D vector back into a dictionary matching the template."""
    res = {}
    idx = 0
    for k, v in template_dict.items():
        numel = v.numel()
        # .clone() ensures we don't accidentally link memory views
        res[k] = vec[idx : idx + numel].view_as(v).clone() 
        idx += numel
    return res

# ==========================================
# --- 2. The Finite-Difference Math Engine ---
# ==========================================

class SharpnessEngine:
    def __init__(self, model, criterion, features, labels):
        self.model = model
        self.criterion = criterion
        self.features = features
        self.labels = labels
        self.buffers = dict(model.named_buffers()) 

    def _compute_loss_with_dict(self, params_dict):
        """Helper to run a functional forward pass with a given parameter dict."""
        predictions = functional_call(self.model, (params_dict, self.buffers), self.features)
        return self.criterion(predictions, self.labels)

    def compute_fd_hvp(self, base_params, v_dict, eps=1e-3, create_graph=False):
        """
        Computes the exact Hessian-Vector Product using Central Finite Difference.
        Crucially, if create_graph=True, the returned HVP maintains a computational 
        graph linked back to `base_params`.
        """
        # 1. Shift parameters slightly forward and backward along eigenvector v
        p_plus = {k: base_params[k] + eps * v_dict[k] for k in base_params.keys()}
        p_minus = {k: base_params[k] - eps * v_dict[k] for k in base_params.keys()}
        
        # 2. Compute losses
        loss_plus = self._compute_loss_with_dict(p_plus)
        loss_minus = self._compute_loss_with_dict(p_minus)
        
        # 3. Compute gradients WITH RESPECT TO THE BASE PARAMETERS
        # Because p_plus = base_params + eps * v, by the chain rule, differentiating 
        # with respect to base_params yields the gradient at the perturbed location.
        grad_plus = torch.autograd.grad(
            loss_plus, base_params.values(), create_graph=create_graph
        )
        grad_minus = torch.autograd.grad(
            loss_minus, base_params.values(), create_graph=create_graph
        )
        
        # 4. Approximate the HVP via Central Difference and repackage into a dict
        hvp_dict = {}
        for i, k in enumerate(base_params.keys()):
            hvp_dict[k] = (grad_plus[i] - grad_minus[i]) / (2 * eps)
            
        return hvp_dict

    def find_top_eigenvector(self, base_params, device):
        """Uses SciPy's Lanczos solver (eigsh) to find the top eigenvector (u)."""
        D = dict_to_vector(base_params).shape[0]
        
        def matvec(v_np):
            # Move SciPy's vector to PyTorch GPU
            v_tensor = torch.from_numpy(v_np).to(device, dtype=torch.float32)
            v_dict = vector_to_dict(v_tensor, base_params)
            
            # Compute HVP without graph tracking (fast)
            hvp_dict = self.compute_fd_hvp(base_params, v_dict, create_graph=False)
            
            # Move back to CPU/NumPy
            return dict_to_vector(hvp_dict).cpu().numpy()

        A = LinearOperator((D, D), matvec=matvec, dtype=np.float32)
        
        # Run Lanczos
        eigenvalues, eigenvectors = eigsh(A, k=1, which='LA', tol=1e-3)
        
        u_vec = torch.from_numpy(eigenvectors[:, 0]).to(device, dtype=torch.float32)
        return vector_to_dict(u_vec, base_params)

    def compute_grad_S(self, base_params, u_dict, eps=1e-3):
        """Calculates Sharpness and the Gradient of Sharpness."""
        
        # 1. Compute HVP WITH graph tracking attached to base_params
        hvp_dict = self.compute_fd_hvp(base_params, u_dict, eps=eps, create_graph=True)
        
        # 2. Compute Sharpness S = u^T (H u). 
        # S is now a scalar tensor with a full autograd history pointing to base_params.
        S = sum(torch.sum(u_dict[k] * hvp_dict[k]) for k in u_dict.keys())
        
        # 3. The 3rd-Order Miracle: We compute \nabla S using standard 2nd-order autograd!
        grad_S_tuple = torch.autograd.grad(S, base_params.values())
        
        # 4. Package it up
        grad_S_dict = {k: g for k, g in zip(base_params.keys(), grad_S_tuple)}
        
        return dict_to_vector(grad_S_dict), S.detach()


# ==========================================
# --- 3. The Integrated Training Loop ---
# ==========================================

def advanced_gd_run(model, X, y, lr, wd, num_epochs, log_every_n=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    features = X.to(device)
    labels = y.to(device)
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    num_tracked = num_epochs // log_every_n
    
    history = {
        'loss': torch.zeros(num_tracked),
        'sharpness': torch.zeros(num_tracked),
        'dot_gradL_gradS': torch.zeros(num_tracked),
        'dot_theta_gradS': torch.zeros(num_tracked),
        'gradL_norm': torch.zeros(num_tracked),
        'theta_norm': torch.zeros(num_tracked),
    }

    log_idx = 0
    engine = SharpnessEngine(model, criterion, features, labels)

    for epoch in range(num_epochs):
        # --- 1. Standard Forward/Backward ---
        predictions = model(features)
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # --- 2. The Heavy Lifting (Logging Step) ---
        if epoch % log_every_n == 0:
            # EXTRACT AND ISOLATE: We clone and detach the current parameters, 
            # and set requires_grad=True. This creates a completely isolated 
            # computational graph purely for tracking S and \nabla S, 
            # ensuring we never corrupt the main training loop's gradients.
            base_params = {k: p.detach().clone().requires_grad_(True) for k, p in model.named_parameters()}
            
            theta_vec = nn_utils.parameters_to_vector(model.parameters()).detach()
            grad_L_vec = nn_utils.parameters_to_vector([p.grad for p in model.parameters()]).detach()
            
            # A. Find the top eigenvector (u) via SciPy Lanczos
            u_dict = engine.find_top_eigenvector(base_params, device)
            
            # B. Compute Sharpness (S) and its gradient (\nabla S) via Finite Difference
            grad_S_vec, sharpness_val = engine.compute_grad_S(base_params, u_dict)
            
            # C. Compute Dot Products
            dot_gL_gS = torch.dot(grad_L_vec, grad_S_vec)
            dot_t_gS = torch.dot(theta_vec, grad_S_vec)
            
            # D. Store safely on CPU
            history['loss'][log_idx] = loss.item()
            history['sharpness'][log_idx] = sharpness_val.item()
            history['dot_gradL_gradS'][log_idx] = dot_gL_gS.item()
            history['dot_theta_gradS'][log_idx] = dot_t_gS.item()
            history['gradL_norm'][log_idx] = grad_L_vec.norm().item()
            history['theta_norm'][log_idx] = theta_vec.norm().item()
            
            log_idx += 1

        # --- 3. Update Weights ---
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("\nTraining complete!")
    return history