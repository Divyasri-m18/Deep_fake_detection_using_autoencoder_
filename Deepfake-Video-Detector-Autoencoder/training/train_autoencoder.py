import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import yaml
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.autoencoder import Autoencoder
from preprocessing.normalize_faces import get_dataloader
from utils.error_analysis import calculate_mse, compute_statistics, save_stats

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_anomaly_detector():
    """
    Trains the Autoencoder strictly on REAL data as an Anomaly Detector.
    """
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')
    print(f"Training on: {device}")
    
    # --- 1. Data Setup (STRICT: REAL ONLY) ---
    kaggle_dir = config['paths']['kaggle_data_path']
    real_faces_dir = None
    
    # Find directory containing 'real' faces
    if os.path.exists(kaggle_dir):
        for root, dirs, files in os.walk(kaggle_dir):
            if 'real' in os.path.basename(root).lower():
                # Verify it has images
                if any(f.endswith(('.jpg', '.png')) for f in files):
                    real_faces_dir = root
                    break
    
    if not real_faces_dir:
        # Check local fallback
        local_real = os.path.join(config['paths']['processed_faces'], 'real')
        if os.path.exists(local_real):
             real_faces_dir = local_real

    if not real_faces_dir:
        print("CRITICAL ERROR: No 'Real' face data found. Anomaly detection requires REAL data training.")
        return

    print(f"Training Data Source (REAL ONLY): {real_faces_dir}")
    
    # Loaders
    batch_size = config['training']['batch_size']
    # Create Split - We need Validation set to calculate Threshold
    # For simplicity, we'll use a single loader and just use a subset for stats later, 
    # OR we can trust get_dataloader to shuffle.
    train_loader = get_dataloader(real_faces_dir, batch_size=batch_size, shuffle=True, is_train=True)
    
    # --- 2. Model Setup ---
    model = Autoencoder().to(device)
    
    # Loss: Reconstruction Error (MSE) is the core metric for Anomaly Detection
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scaler = GradScaler()
    
    epochs = config['training']['epochs']
    save_dir = config['paths']['model_save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 3. Training Loop ---
    print(f"Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images in loop:
            images = images.to(device)
            
            with autocast():
                # Forward
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
             torch.save(model.state_dict(), os.path.join(save_dir, f"autoencoder_epoch_{epoch+1}.pth"))

    # Final Save
    model_path = config['paths']['autoencoder_path']
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # --- 4. Statistical Calibration (THE FIX) ---
    # We must determine what "Normal" error looks like.
    # Pass a subset of REAL videos through the trained model.
    
    print("\n--- Calibrating Threshold (Computing Baseline Stats) ---")
    model.eval()
    all_errors = []
    
    # Use the same loader (or a separate val loader) to check distribution on "Real" data
    calibration_batches = 50 # Check ~3200 faces to get a good distribution
    
    with torch.no_grad():
        for i, images in enumerate(train_loader):
            if i >= calibration_batches: break
            
            images = images.to(device)
            recon = model(images)
            
            # Calculate MSE per image (NOT mean over batch)
            # (N, C, H, W) -> (N,)
            batch_errors = calculate_mse(images, recon)
            all_errors.extend(batch_errors)
            
    # Compute Stats
    stats = compute_statistics(all_errors)
    print(f"Baseline Stats (Real Data):")
    print(f"  Mean Error: {stats['mean']:.6f}")
    print(f"  Std Dev:    {stats['std']:.6f}")
    print(f"  Max Error:  {stats['max']:.6f}")
    
    # Save Stats
    stats_path = os.path.join(save_dir, "model_stats.json")
    save_stats(stats, stats_path)
    print(f"Statistics saved to {stats_path}")
    print("Training & Calibration Complete.")

if __name__ == "__main__":
    train_anomaly_detector()
