import torch
import torch.nn as nn
from monai.networks.nets import resnet18

class TrueHybrid3D(nn.Module):
    def __init__(self, resnet_path: str):
        super().__init__()
        
        # === BRANCH 1: Frozen ResNet-18 ===
        self.resnet = resnet18(
            spatial_dims=3, 
            n_input_channels=1, 
            num_classes=512
        )
        # Load pretrained weights
        state_dict = torch.load(resnet_path, map_location="cpu")
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("resnet."):
                new_sd[k.replace("resnet.", "", 1)] = v
        self.resnet.load_state_dict(new_sd, strict=False)
        # FREEZE all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # === BRANCH 2: Lightweight Dense-Connected 3D CNN ===
        self.dense_branch = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # Block 2 (dense connection)
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # Block 3
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.dense_fc = nn.Linear(32, 256)
        
        # === BRANCH 3: SE-Block Attention 3D CNN ===
        # Smaller version of HybridAttention3DCNN
        self.attn_branch = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.attn_fc = nn.Linear(64, 256)
        
        # === FUSION: Learned Attention ===
        # Concatenated input: 512 + 256 + 256 = 1024
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 3),      # 3 weights (one per branch)
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Linear(1024, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: Frozen ResNet (no gradient)
        with torch.no_grad():
            f_res = self.resnet(x)  # (B, 512)
        
        # Branch 2: Dense-connected
        f_dense = self.dense_branch(x)
        f_dense = f_dense.view(f_dense.size(0), -1)
        f_dense = self.dense_fc(f_dense)  # (B, 256)
        
        # Branch 3: Attention
        f_attn = self.attn_branch(x)
        f_attn = f_attn.view(f_attn.size(0), -1)
        f_attn = self.attn_fc(f_attn)  # (B, 256)
        
        # Concatenate all features
        fused = torch.cat([f_res, f_dense, f_attn], dim=1)  # (B, 1024)
        
        # Learn attention weights
        weights = self.attention(fused)  # (B, 3)
        
        # Apply weights
        w_res = weights[:, 0:1]     # (B, 1)
        w_dense = weights[:, 1:2]   # (B, 1)
        w_attn = weights[:, 2:3]    # (B, 1)
        
        weighted = torch.cat([
            f_res * w_res,
            f_dense * w_dense,
            f_attn * w_attn
        ], dim=1)  # (B, 1024)
        
        return self.classifier(weighted)
