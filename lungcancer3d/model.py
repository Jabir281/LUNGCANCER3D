import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention Module (inspired by CBAM).
    Focuses on 'where' is an informative part in spatial dimensions.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        
        # Ensure kernel_size is odd for proper padding
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Convolve along spatial dimensions after channel pooling
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise pooling: max and avg pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        # Concatenate along channel dimension
        pooled = torch.cat([avg_out, max_out], dim=1)  # (B, 2, D, H, W)
        
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv(pooled))  # (B, 1, D, H, W)
        
        return x * attention

class ChannelAttention3D(nn.Module):
    """
    3D Channel Attention Module (SE-Block adapted for 3D).
    Focuses on 'what' is meaningful by emphasizing important feature channels.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention3D, self).__init__()
        
        # Adaptive average pooling to reduce spatial dimensions to 1x1x1
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        
        # Max pooling path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        
        return x * attention

class LungCancer3DCNN(nn.Module):
    """
    3D CNN with Channel and Spatial Attention for Lung Cancer Detection.
    
    Input: (Batch, 1, 64, 64, 64)
    Output: (Batch, 1) - single logit for binary classification
    """
    
    def __init__(self, in_channels=1, dropout_rate=0.3):
        super(LungCancer3DCNN, self).__init__()
        
        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 64 -> 32
        )
        
        # Second convolution block with Channel Attention
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.channel_attn1 = ChannelAttention3D(64, reduction_ratio=8)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 32 -> 16
        
        # Third convolution block with Spatial Attention
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.spatial_attn = SpatialAttention3D(kernel_size=7)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 16 -> 8
        
        # Fourth convolution block with Channel Attention
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.channel_attn2 = ChannelAttention3D(256, reduction_ratio=16)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # 8 -> 4
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1)  # Single logit output
        )
    
    def forward(self, x):
        # Forward pass through convolution blocks
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.channel_attn1(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.spatial_attn(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.channel_attn2(x)
        x = self.pool4(x)
        
        # Global pooling and fully connected layers
        x = self.global_avg_pool(x)
        x = self.fc(x)
        
        return x.squeeze(-1)  # (Batch, 1) -> (Batch,)

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    model = LungCancer3DCNN()
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 64, 64, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
