import torch
import numpy as np

def generate_linear_data(num_samples: int, input_dim: int, noise_std: float = 0.1, true_weights: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic linear data with optional Gaussian noise."""
    X = torch.randn(num_samples, input_dim)
    if true_weights is None:
        true_weights = torch.normal(1, 2, (input_dim, 1)) # Random weights with mean=1 and std=2
    y = torch.matmul(X, true_weights) + noise_std * torch.randn(num_samples, 1)
    return X, y

def generate_ill_conditioned_data(num_samples, input_dim, noise_std=5):
    X = torch.randn(num_samples, input_dim)
    
    # Create a massive disparity in feature scales (e.g., [1, 10, 100, ...])
    # This creates a "ravine" in the loss landscape
    scales = torch.tensor([2.0 ** i for i in range(input_dim)])
    X = X * scales
    
    true_weights = torch.randn(input_dim, 1)
    
    # Add a LOT of noise. Overfitting to noise drives up sharpness.
    noise = torch.normal(0, noise_std, (num_samples, 1)) 
    y = torch.matmul(X, true_weights) + noise
    
    return X, y

class LinearModel(torch.nn.Module):
    def __init__(self, input_dim: int, d: int, depth: int):
        super().__init__()
        # Use a list comprehension to create distinct layers
        full_layers = [torch.nn.Linear(input_dim, d)] + \
                      [torch.nn.Linear(d, d) for _ in range(depth)] + \
                      [torch.nn.Linear(d, 1)]
        self.linear = torch.nn.Sequential(*full_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class FullReLU(torch.nn.Module):
    def __init__(self, input_dim: int, d: int, depth: int):
        super().__init__()
        full_layers = [torch.nn.Linear(input_dim, d), torch.nn.ReLU()]
        
        # Safely extend the list with fresh instances
        for _ in range(depth):
            full_layers.extend([torch.nn.Linear(d, d), torch.nn.ReLU()])
            
        full_layers.append(torch.nn.Linear(d, 1))
        self.linear = torch.nn.Sequential(*full_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)