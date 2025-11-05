# src/qca_operator_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Bloque de Convolución de Ayuda ---
class ConvBlock(nn.Module):
    """(Conv => GroupNorm => ELU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # Ajustado para manejar casos donde los canales son menores que 8
        num_groups_mid = max(1, mid_channels // 8)
        if mid_channels % num_groups_mid != 0:
             num_groups_mid = 1 # Fallback si no es divisible
             
        num_groups_out = max(1, out_channels // 8)
        if out_channels % num_groups_out != 0:
             num_groups_out = 1 # Fallback
             
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

# --- El Operador U-Net ---
class QCA_Operator_UNet(nn.Module):
    """
    Arquitectura U-Net con 4 NIVELES de bajada/subida.
    Todas las skip connections y canales están CORREGIDOS.
    """
    def __init__(self, d_state, hidden_channels):
        super().__init__()
        self.d_state = d_state
        base_c = hidden_channels # ej: 64
        in_c = 2 * d_state 

        # --- Encoder (Contracción) ---
        self.inc = ConvBlock(in_c, base_c)           # Canales: in_c -> 64 (x1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c, base_c * 2))     # Canales: 64 -> 128 (x2)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 2, base_c * 4)) # Canales: 128 -> 256 (x3)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 4, base_c * 8)) # Canales: 256 -> 512 (x4)

        # --- Bottleneck ---
        self.bot = ConvBlock(base_c * 8, base_c * 16) # Canales: 512 -> 1024

        # --- Decoder (Expansión) ---
        
        # Nivel 1 (sube de H/8 a H/4)
        self.up1 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        # Canales In: up1(base_c*8) + skip x4(base_c*8) = base_c*16
        self.conv_up1 = ConvBlock(base_c * 16, base_c * 8) 
        
        # Nivel 2 (sube de H/4 a H/2)
        self.up2 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        # Canales In: up2(base_c*4) + skip x3(base_c*4) = base_c*8
        self.conv_up2 = ConvBlock(base_c * 8, base_c * 4) 
        
        # Nivel 3 (sube de H/2 a H)
        self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        # Canales In: up3(base_c*2) + skip x2(base_c*2) = base_c*4
        self.conv_up3 = ConvBlock(base_c * 4, base_c * 2) 
        
        # Nivel 4 (sube de H a H - ¡ERROR ANTERIOR ESTABA AQUÍ!)
        # Debería ser de H/2 a H. Y la anterior de H/4 a H/2.
        # Mi lógica de "4 niveles" era confusa.
        # Corrijamos a la U-Net de 4 niveles estándar (inc + 3 down, 3 up + outc)
        
        # --- RE-RE-CORRECCIÓN: __init__ con 3 niveles de bajada ---
        # Esto es más estándar y menos propenso a errores.
        self.inc_b = ConvBlock(in_c, base_c)           # x1 (H)
        self.down1_b = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c, base_c * 2))     # x2 (H/2)
        self.down2_b = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 2, base_c * 4)) # x3 (H/4)
        self.down3_b = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 4, base_c * 8)) # x4 (H/8)

        # Bottleneck
        self.bot_b = ConvBlock(base_c * 8, base_c * 8) # (H/8) -> (H/8)

        # Decoder
        self.up1_b = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        # Canales In: up1(base_c*4) + skip x3(base_c*4) = base_c*8
        self.conv_up1_b = ConvBlock(base_c * 8, base_c * 4) # (H/4)

        self.up2_b = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        # Canales In: up2(base_c*2) + skip x2(base_c*2) = base_c*4
        self.conv_up2_b = ConvBlock(base_c * 4, base_c * 2) # (H/2)

        self.up3_b = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        # Canales In: up3(base_c) + skip x1(base_c) = base_c*2
        self.conv_up3_b = ConvBlock(base_c * 2, base_c) # (H)

        # Salida
        self.outc_b = nn.Conv2d(base_c, 2 * d_state, kernel_size=1)
        
    def forward(self, x_cat):
        # x_cat tiene forma [B, 2*d_state, H, W]
        
        # --- Encoder (3 niveles) ---
        x1 = self.inc_b(x_cat)  # -> [B, base_c, H, W]
        x2 = self.down1_b(x1)   # -> [B, base_c*2, H/2, W/2]
        x3 = self.down2_b(x2)   # -> [B, base_c*4, H/4, W/4]
        x4 = self.down3_b(x3)   # -> [B, base_c*8, H/8, W/8]

        # --- Bottleneck ---
        b = self.bot_b(x4)      # -> [B, base_c*8, H/8, W/8]

        # --- Decoder con Skip Connections (CORREGIDO) ---
        u1 = self.up1_b(b)                                # -> [B, base_c*4, H/4, W/4]
        s1 = torch.cat([u1, x3], dim=1)                 # Concat skip: [B, base_c*8, H/4, W/4]
        c1 = self.conv_up1_b(s1)                          # -> [B, base_c*4, H/4, W/4]

        u2 = self.up2_b(c1)                               # -> [B, base_c*2, H/2, W/2]
        s2 = torch.cat([u2, x2], dim=1)                 # Concat skip: [B, base_c*4, H/2, W/2]
        c2 = self.conv_up2_b(s2)                          # -> [B, base_c*2, H/2, W/2]
        
        u3 = self.up3_b(c2)                               # -> [B, base_c, H, W]
        s3 = torch.cat([u3, x1], dim=1)                 # Concat skip: [B, base_c*2, H, W]
        c3 = self.conv_up3_b(s3)                          # -> [B, base_c, H, W]

        # --- Capa de Salida ---
        output = self.outc_b(c3) # Forma: [B, 2*d_state, H, W]

        # --- Reformatear a la salida esperada por Aetheria_Motor ---
        output = output.permute(0, 2, 3, 1) # (B, H, W, 2*d_state)
        output = output.squeeze(0) # Forma: [H, W, 2*d_state]
        
        # Dividir en real e imag
        delta_real = output[..., :self.d_state]
        delta_imag = output[..., self.d_state:]
        
        # Aplicar la misma escala que tenías en tu modelo original
        delta_real = delta_real * 0.1
        delta_imag = delta_imag * 0.1

        return delta_real, delta_imag