import numpy as np
import matplotlib.pyplot as plt
import json
import os

def calculate_mse(input_tensor, reconstruction_tensor):
    """
    Calculates Mean Squared Error between input and reconstruction.
    Args:
        input_tensor: (N, C, H, W) or (C, H, W)
        reconstruction_tensor: (N, C, H, W) or (C, H, W)
    Returns:
        mse: scalar or array of shape (N,)
    """
    # specific implementation depends on tensor vs numpy
    # Here assuming numpy for final aggregation or torch for batch
    import torch
    
    if isinstance(input_tensor, torch.Tensor):
        with torch.no_grad():
            diff = input_tensor - reconstruction_tensor
            # Mean over pixels (C, H, W) -> Result is (N,) or scalar
            mse = torch.mean(diff ** 2, dim=[1, 2, 3])
            return mse.cpu().numpy()
            
    # Numpy fallback
    diff = input_tensor - reconstruction_tensor
    mse = np.mean(diff ** 2, axis=(1, 2, 3))
    return mse

def compute_statistics(errors):
    """
    Computes Mean, Std, Min, Max of a list of errors.
    """
    errors = np.array(errors)
    stats = {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "percentile_95": float(np.percentile(errors, 95)),
        "percentile_99": float(np.percentile(errors, 99))
    }
    return stats

def get_threshold(stats, k=2.5, method="std"):
    """
    Calculates the anomaly threshold.
    Args:
        stats: dict from compute_statistics (must have 'mean', 'std')
        k: multiplier for std dev
        method: "std" or "percentile"
    """
    if method == "percentile":
        return stats["percentile_95"]
    
    # Default: Mean + k * Std
    threshold = stats["mean"] + (k * stats["std"])
    return threshold

def save_stats(stats, path):
    with open(path, 'w') as f:
        json.dump(stats, f, indent=4)

def load_stats(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def plot_error_distribution(errors, threshold, save_path=None):
    """
    Plots a histogram of reconstruction errors and marks the threshold.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='blue', label='Reconstruction Errors')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MSE Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() # In CLI mode this might not show, better to save
