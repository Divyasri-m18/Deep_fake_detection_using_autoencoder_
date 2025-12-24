import torch
import numpy as np

def compute_reconstruction_error(faces_tensor, model, device):
    """
    Computes MSE reconstruction error for a batch of faces.
    Args:
        faces_tensor: Tensor of shape (B, 3, H, W)
        model: Trained Autoencoder
        device: torch device
    Returns:
        errors: np array of shape (B,) containing MSE per image
    """
    model.eval()
    with torch.no_grad():
        faces_tensor = faces_tensor.to(device)
        reconstructed = model(faces_tensor)
        
        # Calculate MSE per image
        # (B, 3, H, W)
        mse = (faces_tensor - reconstructed) ** 2
        
        # Mean over 3, H, W
        mse = mse.mean(dim=[1, 2, 3])
        
        return mse.cpu().numpy()
