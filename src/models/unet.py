# src/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model

# --- Bloque de Convolución de Ayuda ---
class ConvBlock(nn.Module):
    """(Conv => GroupNorm => ELU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # --- ¡¡AQUÍ ESTÁ LA CORRECCIÓN!! ---
        
        # El número de grupos para la PRIMERA capa GroupNorm
        # debe ser un divisor de 'mid_channels'
        num_groups_mid = max(1, mid_channels // 8)
        if mid_channels % num_groups_mid != 0:
             # Fallback si 8 no es un divisor (ej. 64 % 8 sí, pero 21 % 8 no)
             # Buscamos el divisor más grande posible <= 8
             divisors = [8, 4, 2, 1]
             for d in divisors:
                 if mid_channels % d == 0:
                     num_groups_mid = d
                     break
             
        # El número de grupos para la SEGUNDA capa GroupNorm
        # debe ser un divisor de 'out_channels'
        num_groups_out = max(1, out_channels // 8)
        if out_channels % num_groups_out != 0:
             divisors = [8, 4, 2, 1]
             for d in divisors:
                 if out_channels % d == 0:
                     num_groups_out = d
                     break
        # ------------------------------------
             
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_mid, num_channels=mid_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_out, num_channels=out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

# --- El Operador U-Net (Versión 3-Niveles / 7.7M params) ---
@register_model('UNET')
class QCA_Operator_UNet(nn.Module):
    """
    Arquitectura U-Net con 3 NIVELES de bajada/subida (inc + 2 down).
    Esta es la versión estable de ~7.7M de parámetros (con base_c=64).
    """
    def __init__(self, d_state, hidden_channels):
        super().__init__()
        self.d_state = d_state
        base_c = hidden_channels # ej: 64
        in_c = 2 * d_state 

        # --- Encoder (Contracción) ---
        self.inc = ConvBlock(in_c, base_c)           # x1 (H) -> 64
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c, base_c * 2))     # x2 (H/2) -> 128
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 2, base_c * 4)) # x3 (H/4) -> 256
        
        # --- Bottleneck ---
        self.bot = ConvBlock(base_c * 4, base_c * 8) # (H/4) -> 512

        # --- Decoder (Expansión) ---
        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2) # Sube a H/2
        # Canales In: up1(base_c*4) + skip x2(base_c*2) = base_c*6 = 384
        self.conv_up1 = ConvBlock(base_c * 6, base_c * 4) # (H/2) -> 256

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2) # Sube a H
        # Canales In: up2(base_c*2) + skip x1(base_c) = base_c*3 = 192
        self.conv_up2 = ConvBlock(base_c * 3, base_c) # (H) -> 64

        # --- Salida ---
        self.outc = nn.Conv2d(base_c, 2 * d_state, kernel_size=1)
        
    def forward(self, x_cat):
        # x_cat tiene forma [B, 2*d_state, H, W]
        
        # --- Encoder ---
        x1 = self.inc(x_cat)  # [B, 64, H=256]
        x2 = self.down1(x1) # [B, 128, H=128]
        x3 = self.down2(x2) # [B, 256, H=64]
        
        b = self.bot(x3)    # [B, 512, H=64]

        # --- Decoder (¡¡CONEXIONES CORREGIDAS!!) ---
        u1 = self.up1(b)                                # [B, 256, H=128]
        s1 = torch.cat([u1, x2], dim=1)                 # Concat: [B, 256 + 128 = 384, H=128]
        c1 = self.conv_up1(s1)                          # [B, 256, H=128]
        
        u2 = self.up2(c1)                               # [B, 128, H=256]
        s2 = torch.cat([u2, x1], dim=1)                 # Concat: [B, 128 + 64 = 192, H=256]
        c2 = self.conv_up2(s2)                          # [B, 64, H=256]
        
        output = self.outc(c2) # Forma: [B, 2*d_state, H, W]

        # --- Reformatear a la salida esperada por Aetheria_Motor ---
        # The output is already in [B, C, H, W] format, where C = 2*d_state
        # We just need to split and then concatenate again to apply the 0.1 factor
        
        # Permute to [B, H, W, 2*d_state] for splitting
        output_permuted = output.permute(0, 2, 3, 1)
        
        delta_real = output_permuted[..., :self.d_state]
        delta_imag = output_permuted[..., self.d_state:]
        
        delta_real = delta_real * 0.1
        delta_imag = delta_imag * 0.1

        # Permute back to [B, C, H, W] and concatenate
        delta_psi = torch.cat([delta_real.permute(0, 3, 1, 2), delta_imag.permute(0, 3, 1, 2)], dim=1)
        
        return delta_psi