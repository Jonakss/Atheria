import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class RotationalUNet(nn.Module):
    """
    A U-Net specialized for Phase/Magnitude processing.
    Input: 3 Channels (Magnitude, Sin(Theta), Cos(Theta))
    Output: 2 Channels (Delta Magnitude, Delta Theta)
    """
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(RotationalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))

        # Bottleneck
        self.bot = DoubleConv(64, 128)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(128 + 32, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(64 + 16, 32)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Assuming input is divisible by 8?
        # Wait, I only did 2 downs. Let's match up.
        # Down: 128 -> 64 -> 32
        # Up: 32 -> 64 -> 128

        # Output
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

        # No up3 if only 2 downs.
        # Input -> Inc (128) -> Down1 (64) -> Down2 (32) -> Bot (32) -> Up1 (64) -> Up2 (128) -> Out

    def forward(self, x):
        x1 = self.inc(x) # 16
        x2 = self.down1(x1) # 32
        x3 = self.down2(x2) # 64

        x_bot = self.bot(x3) # 128

        x_up1 = self.up1(x_bot)
        # diffY = x2.size()[2] - x_up1.size()[2]
        # diffX = x2.size()[3] - x_up1.size()[3]
        # x_up1 = F.pad(x_up1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Assuming power of 2 size
        x_cat1 = torch.cat([x2, x_up1], dim=1)
        x_dec1 = self.conv_up1(x_cat1)

        x_up2 = self.up2(x_dec1)
        x_cat2 = torch.cat([x1, x_up2], dim=1)
        x_dec2 = self.conv_up2(x_cat2)

        logits = self.outc(x_dec2)

        # Use Tanh to constrain predictions
        # Channel 0: Delta Mag (Additive/Log-space) -> Tanh allows +/- changes
        # Channel 1: Delta Phase (Rotational) -> Tanh * PI allows full rotation range (well, PI/1 scale)
        return torch.tanh(logits)
