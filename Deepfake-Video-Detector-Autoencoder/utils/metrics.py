import torch
import torch.nn.functional as F
import numpy as np

def calculate_mse(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculates Mean Squared Error between output and target.
    Expects tensors of shape (N, C, H, W) or (C, H, W).
    """
    with torch.no_grad():
        loss = F.mse_loss(output, target, reduction='mean')
    return loss.item()

def get_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute MSE between two images (numpy arrays).
    Normalized to 0-1 range usually if images are 0-1.
    """
    return np.mean((original - reconstructed) ** 2)

def is_fake(error: float, threshold: float) -> bool:
    """
    Simple thresholding logic.
    If error > threshold, it's likely an anomaly (Fake), 
    assuming the model is trained well on Real faces.
    """
    return error > threshold
