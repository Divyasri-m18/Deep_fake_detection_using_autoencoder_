import torch
import cv2
import numpy as np
from models.autoencoder import Autoencoder
from torchvision import transforms
from PIL import Image

def debug_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Autoencoder().to(device)
    
    path = "models/trained/autoencoder.pth"
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    # Create a dummy "Real" face (smooth gradient) vs Random Noise
    # A real model should reconstruct smooth things better than random noise.
    
    # 1. Smooth Image (approximating a face/structure)
    x = np.linspace(0, 1, 128)
    y = np.linspace(0, 1, 128)
    xv, yv = np.meshgrid(x, y)
    smooth_img = (xv + yv) / 2.0
    smooth_img = np.stack([smooth_img]*3, axis=2) # (128, 128, 3)
    smooth_img = (smooth_img * 255).astype(np.uint8)
    
    # 2. Random Noise
    noise_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    transform = transforms.Compose([
        transforms.ToTensor(), # [0-1]
    ])
    
    img1 = transform(Image.fromarray(smooth_img)).unsqueeze(0).to(device)
    img2 = transform(Image.fromarray(noise_img)).unsqueeze(0).to(device)
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        out1 = model(img1)
        out2 = model(img2)
        
        loss1 = criterion(out1, img1).item()
        loss2 = criterion(out2, img2).item()
        
    print(f"MSE for Smooth Input: {loss1:.5f}")
    print(f"MSE for Noise Input:  {loss2:.5f}")
    
    if loss1 > 0.05:
        print("WARNING: High reconstruction error on smooth input. Model might be underfit.")
    
    if loss1 >= loss2:
        print("CRITICAL: Model is not distinguishing structure from noise.")
    else:
        print("OK: Model reconstructs structure better than noise.")

if __name__ == "__main__":
    debug_model()
