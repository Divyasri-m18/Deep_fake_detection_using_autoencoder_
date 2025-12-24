import torch
import torch.nn as nn
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import Autoencoder
from preprocessing.normalize_faces import get_dataloader

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_on_set(dataloader, model, device, desc="Evaluating"):
    model.eval()
    criterion = nn.MSELoss(reduction='none') # want per-image loss usually, but batch is fine
    total_loss = 0.0
    all_losses = []
    
    with torch.no_grad():
        for images in tqdm(dataloader, desc=desc):
            images = images.to(device)
            outputs = model(images)
            
            # Per-element MSE
            loss = criterion(outputs, images)
            
            # Mean over pixels (C, H, W)
            # shape: [B, C, H, W] -> [B]
            img_losses = loss.mean(dim=[1, 2, 3])
            
            all_losses.extend(img_losses.cpu().numpy())
            
    return np.mean(all_losses), all_losses

def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')
    
    model = Autoencoder().to(device)
    model_path = config['paths']['autoencoder_path']
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded.")
    
    # Eval Real
    real_dir = os.path.join(config['paths']['processed_faces'], 'real')
    if os.path.exists(real_dir):
        real_loader = get_dataloader(real_dir, batch_size=32, shuffle=False)
        mean_real, losses_real = evaluate_on_set(real_loader, model, device, "Eval Real")
        print(f"Mean Reconstruction Error (Real): {mean_real:.6f}")
        print(f"Max Error (Real): {np.max(losses_real):.6f}")
    
    # Eval Fake
    fake_dir = os.path.join(config['paths']['processed_faces'], 'fake')
    if os.path.exists(fake_dir) and len(os.listdir(fake_dir)) > 0:
        fake_loader = get_dataloader(fake_dir, batch_size=32, shuffle=False)
        mean_fake, losses_fake = evaluate_on_set(fake_loader, model, device, "Eval Fake")
        print(f"Mean Reconstruction Error (Fake): {mean_fake:.6f}")
        print(f"Min Error (Fake): {np.min(losses_fake):.6f}")
        
    # Heuristic for threshold
    if os.path.exists(real_dir) and os.path.exists(fake_dir):
        # suggested threshold could be mean_real + std_real
        suggested_threshold = np.mean(losses_real) + 2 * np.std(losses_real)
        print(f"Suggested Threshold: {suggested_threshold:.6f}")

if __name__ == "__main__":
    main()
