import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    """
    Volumetric U-Net for Atheria 5 (37 Dimensions).
    Processes 5D tensors: (Batch, Channels, Depth, Height, Width).
    
    Architecture:
    - Encoder: Conv3d -> MaxPool3d
    - Bottleneck
    - Decoder: Upsample -> Concat -> Conv3d
    """
    def __init__(self, in_channels: int = 37, out_channels: int = 37, features: int = 32):
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.enc1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 2, features * 4)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = self._block((features * 2) * 2, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = self._block((features) * 2, features)
        
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=features), # GroupNorm is robust for small batches in 3D
            nn.GELU(),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=features),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, D, H, W)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        # Decoder
        # Concatenation requires skip connection resizing if dimensions don't match exactly 
        # (due to odd input sizes), but we assume power-of-2 grids for Atheria.
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

if __name__ == "__main__":
    # Quick debugging test
    model = UNet3D(in_channels=37, out_channels=37)
    x = torch.randn(1, 37, 32, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert x.shape == y.shape
