import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # --- Encoder (EfficientNet-B0) ---
        # We use EfficientNet-B0 as a fixed feature extractor or fine-tunable encoder.
        # EfficientNet-B0 input: (3, H, W). 
        # For 128x128 input, the final feature map is (1280, 4, 4).
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Remove the classification head (fc)
        # We want the features from `extract_features` which returns (N, 1280, H/32, W/32)
        # For 128x128 -> (N, 1280, 4, 4)
        
        # --- Decoder ---
        # Needs to upsample from (1280, 4, 4) back to (3, 128, 128)
        self.decoder = nn.Sequential(
            # Input: 1280 x 4 x 4
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1), # -> 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # -> 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 3, kernel_size=3, padding=1), # Final alignment
            nn.Sigmoid() # Output [0, 1]
        )

    def forward(self, x):
        # x is (N, 3, 128, 128)
        
        # Encoder features
        features = self.encoder.extract_features(x) # -> (N, 1280, 4, 4)
        
        # Decode
        reconstruction = self.decoder(features)
        return reconstruction

if __name__ == "__main__":
    # Test stub
    model = Autoencoder()
    test_input = torch.randn(2, 3, 128, 128)
    output = model(test_input)
    print(f"Input: {test_input.shape}")
    print(f"Output: {output.shape}")
