import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class DeepfakeClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeClassifier, self).__init__()
        
        # Load Pretrained EfficientNet
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b0')
            
        # Replace final classification layer
        # EfficientNet-B0's _fc input features are 1280
        num_ftrs = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Test
    model = DeepfakeClassifier(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(output)
