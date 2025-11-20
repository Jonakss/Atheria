# src/models/unet.py
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv => GroupNorm => ELU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        num_groups_mid = max(1, mid_channels // 8 if mid_channels > 8 else 1)
        if mid_channels % num_groups_mid != 0: num_groups_mid = 1
             
        num_groups_out = max(1, out_channels // 8 if out_channels > 8 else 1)
        if out_channels % num_groups_out != 0: num_groups_out = 1
             
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_mid, num_channels=mid_channels),
            nn.ELU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_out, num_channels=out_channels),
            nn.ELU(inplace=False),
        )

    def forward(self, x):
        return self.conv_block(x)


class UNet(nn.Module):
    _compiles = False  # Desactivar torch.compile para este modelo (problemas con CUDA Graphs)
    """ Arquitectura U-Net estándar. """
    def __init__(self, d_state, hidden_channels, **kwargs):
        super().__init__()
        self.d_state = d_state
        base_c = hidden_channels
        in_c = 2 * d_state
        out_c = 2 * d_state

        self.inc = ConvBlock(in_c, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 2, base_c * 4))
        
        self.bot = ConvBlock(base_c * 4, base_c * 8)

        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(base_c * 6, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(base_c * 3, base_c)

        self.outc = nn.Conv2d(base_c, out_c, kernel_size=1)
        nn.init.normal_(self.outc.weight, mean=0.0, std=1e-5)
        
    def forward(self, x_cat):
        x1 = self.inc(x_cat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        b = self.bot(x3)
        u1 = self.up1(b)
        s1 = torch.cat([u1, x2], dim=1)
        c1 = self.conv_up1(s1)
        u2 = self.up2(c1)
        s2 = torch.cat([u2, x1], dim=1)
        c2 = self.conv_up2(s2)
        delta_psi = self.outc(c2)
        return delta_psi
