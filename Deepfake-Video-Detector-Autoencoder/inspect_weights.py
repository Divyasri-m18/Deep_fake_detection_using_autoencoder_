import torch
import os

path = "models/trained/xception_deepfake.pth"

if not os.path.exists(path):
    print("File not found yet.")
else:
    try:
        state_dict = torch.load(path, map_location='cpu')
        print("Keys found:", list(state_dict.keys())[:5])
        
        # Check specific key format
        if 'model.conv1.weight' in state_dict:
            print("Prefix: model.")
        elif 'conv1.weight' in state_dict:
            print("Prefix: None")
        
        # Check final layer size
        # Usually fc.weight or last_linear.weight
        for k in list(state_dict.keys())[-5:]:
            print(f"{k}: {state_dict[k].shape}")
            
    except Exception as e:
        print(f"Error loading: {e}")
