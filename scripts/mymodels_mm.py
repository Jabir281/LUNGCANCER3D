import torch
import torch.nn as nn
from monai.networks.nets import resnet50

class ResNet50_MLP(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        
        # =====================================================================
        # BACKBONE: 3D ResNet-50
        # =====================================================================
        # Initialize MONAI's 3D ResNet-50
        # Note: MONAI's resnet50 does not natively take a 'pretrained' kwarg.
        # If you need pretraining, you would load the state dict manually here.
        self.backbone = resnet50(
            spatial_dims=3, 
            n_input_channels=1, 
            num_classes=1000 # Dummy size, bypassed below
        )
        
        # Dynamically extract the number of incoming features (2048 for ResNet-50)
        self.in_features = self.backbone.fc.in_features
        
        # Strip the native classification head by replacing it with an Identity layer
        self.backbone.fc = nn.Identity()
        
        # =====================================================================
        # CLASSIFIER: Standardized MLP Head
        # =====================================================================
        # Project raw feature maps down to a single raw logit
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1) # Output: 1 raw logit for BCEWithLogitsLoss
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pass input volume through the 3D backbone
        features = self.backbone(x)
        
        # 2. Handle potential tuple outputs from MONAI backbones (safeguard)
        if isinstance(features, tuple): 
            features = features[-1]
            
        # 3. Flatten the spatial dimensions (B, Features, 1, 1, 1) -> (B, Features)
        features = torch.flatten(features, 1)
        
        # 4. Pass the flattened 1D embeddings through the standardized MLP
        logits = self.classifier(features)
        
        return logits

# =====================================================================
# Example Instantiation & Forward Pass Test
# =====================================================================
if __name__ == "__main__":
    # Initialize the model
    model = ResNet50_MLP(pretrained=False)
    
    # Create a dummy batch of 2 3D volumes: (Batch, Channel, D, H, W)
    dummy_input = torch.randn(2, 1, 64, 64, 64)
    
    # Run the forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [2, 1]