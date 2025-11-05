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
            
        # Usamos GroupNorm en lugar de BatchNorm porque no depende del tamaño del batch.
        # Es mucho más estable para este tipo de simulaciones.
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, mid_channels // 8), num_channels=mid_channels), # Grupos dinámicos
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

# --- El Operador U-Net ---
class QCA_Operator_UNet(nn.Module):
    """
    Reemplazo de QCA_Operator_Deep con una arquitectura U-Net.
    Es 100% convolucional y agnóstico al tamaño de la cuadrícula.
    
    Toma `x_cat` [B, 2*d_state, H, W] como entrada.
    Saca `delta_real` y `delta_imag` [H, W, d_state] como salida.
    """
    def __init__(self, d_state, hidden_channels):
        super().__init__()
        self.d_state = d_state
        
        # 'hidden_channels' en config.py ahora controla el "ancho" de la U-Net.
        # Un valor como 64 es un buen punto de partida.
        base_c = hidden_channels # ej: 64
        
        in_c = 2 * d_state # Canales de entrada (x_real + x_imag)

        # --- Encoder (Contracción) ---
        self.inc = ConvBlock(in_c, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 2, base_c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 4, base_c * 8))

        # --- Bottleneck ---
        self.bot = ConvBlock(base_c * 8, base_c * 16)

        # --- Decoder (Expansión) ---
        self.up1 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(base_c * 16, base_c * 8) # (skip + up)
        
        self.up2 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(base_c * 8, base_c * 4) # (skip + up)
        
        self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(base_c * 4, base_c * 2) # (skip + up)
        
        self.up4 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.conv_up4 = ConvBlock(base_c * 2, base_c) # (skip + up)

        # Capa de salida: debe producir 2 * d_state canales (para real e imag)
        self.outc = nn.Conv2d(base_c, 2 * d_state, kernel_size=1)
        
        # Nota: La U-Net tiene 4 niveles de bajada/subida.
        # Si tu cuadrícula de entrenamiento es 256, baja a: 128, 64, 32, 16.
        # Si entrenas con cuadrículas más pequeñas (ej. 32),
        # tendrías que reducir el número de capas 'down'.
        # Para 256x256, 4 niveles está perfecto.

    def forward(self, x_cat):
        # x_cat tiene forma [B, 2*d_state, H, W]
        
        # --- Encoder ---
        x1 = self.inc(x_cat)  # -> [B, base_c, H, W]
        x2 = self.down1(x1)   # -> [B, base_c*2, H/2, W/2]
        x3 = self.down2(x2)   # -> [B, base_c*4, H/4, W/4]
        x4 = self.down3(x3)   # -> [B, base_c*8, H/8, W/8]

        # --- Bottleneck ---
        b = self.bot(x4)      # -> [B, base_c*16, H/8, W/8]

        # --- Decoder con Skip Connections ---
        u1 = self.up1(b)                                # -> [B, base_c*8, H/4, W/4]
        s1 = torch.cat([u1, x3], dim=1)                 # Concat skip: [B, base_c*16, H/4, W/4]
        c1 = self.conv_up1(s1)                          # -> [B, base_c*8, H/4, W/4]

        u2 = self.up2(c1)                               # -> [B, base_c*4, H/2, W/2]
        s2 = torch.cat([u2, x2], dim=1)                 # Concat skip: [B, base_c*8, H/2, W/2]
        c2 = self.conv_up2(s2)                          # -> [B, base_c*4, H/2, W/2]
        
        u3 = self.up3(c2)                               # -> [B, base_c*2, H, W]
        s3 = torch.cat([u3, x1], dim=1)                 # Concat skip: [B, base_c*4, H, W]
        c3 = self.conv_up3(s3)                          # -> [B, base_c*2, H, W]

        # --- Capa de Salida ---
        # (He quitado una capa 'up' extra que estaba en mi borrador anterior,
        # 3 niveles de bajada y 3 de subida (más 'inc' y 'bot') es más estándar).
        # (RE-corrigiendo: no, 4 niveles está bien para 256. El código anterior estaba bien).
        # (Restaurando el código original de 4 niveles que te di)
        u4 = self.up4(c3)                               # -> [B, base_c, H, W]
        # La U-Net original (3 niveles down) se conectaría con x1.
        # Mi U-Net (4 niveles down) es más profunda.
        # Re-escribo esta parte para que coincida con el encoder de arriba.
        
        # Encoder: x1(H), x2(H/2), x3(H/4), x4(H/8)
        # Bottleneck: b(H/8)
        # Decoder:
        # u1(b) -> H/4. cat(u1, x3). c1 -> H/4
        # u2(c1) -> H/2. cat(u2, x2). c2 -> H/2
        # u3(c2) -> H.   cat(u3, x1). c3 -> H
        
        # (El código anterior u3(c2) y s3=cat(u3,x1) era correcto)
        
        output = self.outc(c3) # Forma: [B, 2*d_state, H, W]

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