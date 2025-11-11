# src/models/unet_unitary.py
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
             divisors = [4, 2, 1]
             for d in divisors:
                 if mid_channels % d == 0:
                     num_groups_mid = d
                     break
             
        # El número de grupos para la SEGUNDA capa GroupNorm
        # debe ser un divisor de 'out_channels'
        num_groups_out = max(1, out_channels // 8)
        if out_channels % num_groups_out != 0:
             divisors = [4, 2, 1]
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
@register_model('UNET_UNITARIA')
class QCA_Operator_UNet_Unitary(nn.Module):
    """
    Arquitectura U-Net con 3 NIVELES de bajada/subida (inc + 2 down).
    Esta es la versión estable de ~7.7M de parámetros (con base_c=64).
    """
    def _initialize_weights(self):
        """Inicializa los pesos de la capa de salida para que sean muy pequeños."""
        print("Inicializando capa de salida (outc) con pesos pequeños.")
        nn.init.normal_(self.outc.weight, mean=0.0, std=1e-5)
    
    
    def __init__(self, d_vector, hidden_channels):
        super().__init__()
        self.d_vector = d_vector # ej. 4
        base_c = hidden_channels     # ej. 64
        in_c = 2 * d_vector              # ej. 4 <--- CORREGIDO
        
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
        self.outc = nn.Conv2d(base_c, self.d_vector * self.d_vector, kernel_size=1)
        
        # --- ¡¡NUEVA LÍNEA!! ---
        self._initialize_weights() # Asegura que el delta inicial sea pequeño
        
    def forward(self, x_cat):
        # x_cat (psi_t) tiene forma [B, C, H, W] (ej: [1, 42, 256, 256])
        B, C, H, W = x_cat.shape 
        
        # 1. Convertir x_cat a representación compleja
        # x_cat es [B, 2*D_STATE, H, W]
        # d_vector es D_STATE
        x_real = x_cat[:, :self.d_vector, :, :] # [B, D_STATE, H, W]
        x_imag = x_cat[:, self.d_vector:, :, :] # [B, D_STATE, H, W]
        complex_psi = torch.complex(x_real, x_imag) # [B, D_STATE, H, W]

        # --- 1. U-Net ---
        # La U-Net opera sobre la representación real-valorada x_cat
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
        
        # --- 2. Predecir la Matriz 'A' ---
        # A_raw tiene forma [B, D*D, H, W]
        A_raw = self.outc(c2) 
        
        # --- 3. Reformatear A ---
        # [B, D*D, H, W] -> [B, H, W, D, D]
        A_raw = A_raw.permute(0, 2, 3, 1).view(B, H, W, self.d_vector, self.d_vector)
        
        # --- 4. Forzar Anti-Simetría (A = -A.T) ---
        # A shape: [B, H, W, D, D]
        A = 0.5 * (A_raw - A_raw.transpose(-1, -2)) 
        
        # Convertir A a tipo complejo con parte imaginaria cero
        A_complex = A.to(complex_psi.dtype) # Convert A to ComplexFloat
        
        # --- 5. Calcular el Delta (dΨ/dt = A * Ψ) usando matmul ---
        # A_complex (bxyij):      [B, H, W, D, D] (ahora complejo)
        # complex_psi (bjxy):   [B, D, H, W] (complejo)

        # Permutar complex_psi to [B, H, W, D] for matmul
        complex_psi_permuted = complex_psi.permute(0, 2, 3, 1) # [B, H, W, D]

        # Perform batch matrix-vector multiplication
        # Result will be [B, H, W, D]
        delta_complex_psi_permuted = torch.matmul(A_complex, complex_psi_permuted.unsqueeze(-1)).squeeze(-1)

        # Permute back to [B, D, H, W]
        delta_complex_psi = delta_complex_psi_permuted.permute(0, 3, 1, 2)
        
        # 6. Convertir delta_complex_psi de nuevo a representación real-valorada
        delta_psi_real = delta_complex_psi.real
        delta_psi_imag = delta_complex_psi.imag
        delta_psi = torch.cat([delta_psi_real, delta_psi_imag], dim=1) # [B, 2*D_STATE, H, W]
        
        # --- 7. Devolver el Delta ---
        return delta_psi