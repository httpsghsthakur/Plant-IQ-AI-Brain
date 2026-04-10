import torch
import torch.nn as nn
from torchvision import models

class PlantDiseaseCNN(nn.Module):
    """
    A high-performance Vision model based on the ShuffleNetV2 architecture.
    Optimized for 'Edge AI' (Farm tablets) to provide high accuracy with 
    extremely low latency.
    """
    def __init__(self, num_classes=5):
        super(PlantDiseaseCNN, self).__init__()
        
        # Load the ShuffleNetV2 backbone
        # We use x1.0 for a balance of speed and deep feature extraction
        # Set weights=None initially; weights will be loaded during fine-tuning
        self.backbone = models.shufflenet_v2_x1_0(weights=None)
        
        # Replace the final fully connected layer (linear) to match our plant disease classes
        # ShuffleNetV2 x1.0 uses 1024 output features before the final classifier
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
