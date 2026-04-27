import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

def get_cifar10_subset(num_samples=1000):
    # 1. Standardize the data to mean 0, std 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 2. Extract a small subset for full-batch GD
    subset_indices = torch.randperm(len(dataset))[:num_samples]
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = torch.utils.data.DataLoader(subset, batch_size=num_samples, shuffle=False)
    
    X, y = next(iter(dataloader))
    
    # 3. Flatten images (Batch, Channels, Height, Width) -> (Batch, 3072)
    X = X.view(X.size(0), -1)
    
    # 4. One-hot encode the labels for MSE Loss and cast to float
    y_one_hot = F.one_hot(y, num_classes=10).float()
    
    return X, y_one_hot

# Model Setup:
# input_dim = 3072 (3 * 32 * 32), output_dim = 10
# Example: model = LinearModel(input_dim=3072, d=128, depth=2) 
# Remember to change your LinearModel's final layer to output 10, not 1!