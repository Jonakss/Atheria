# src/qca_operator_unet_unitary.py
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
        in_c = d_vector              # ej. 4
        
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
        # x_cat (psi_t) tiene forma [B, C, H, W] (ej: [1, 4, 256, 256])
        B, C, H, W = x_cat.shape 
        
        # --- 1. U-Net ---
        x1 = self.inc(x_cat)  # [B, 64, H=256]
        x2 = self.down1(x1) # [B, 128, H=128]
        x3 = self.down2(x2) # [B, 256, H=64]
        
        b = self.bot(x3)    # [B, 512, H=64]

        u1 = self.up1(b)                                # [B, 256, H=128]
        s1 = torch.cat([u1, x2], dim=1)                 # Concat: [B, 256 + 128 = 384, H=128]
        c1 = self.conv_up1(s1)                          # [B, 256, H=128]
        
        u2 = self.up2(c1)                               # [B, 128, H=256]
        s2 = torch.cat([u2, x1], dim=1)                 # Concat: [B, 128 + 64 = 192, H=256]
        c2 = self.conv_up2(s2)                          # [B, 64, H=256]
        
        # --- 2. Predecir la Matriz 'A' ---
        # A_raw tiene forma [B, D*D, H, W] (ej: [1, 16, 256, 256])
        A_raw = self.outc(c2) 
        
        # --- 3. Reformatear A ---
        # [B, D*D, H, W] -> [B, H, W, D, D]
        A_raw = A_raw.permute(0, 2, 3, 1).view(B, H, W, self.d_vector, self.d_vector)
        
        # --- 4. Forzar Anti-Simetría (A = -A.T) ---
        # A shape: [B, H, W, D, D] (ej: [1, 256, 256, 4, 4])
        A = 0.5 * (A_raw - A_raw.transpose(-1, -2)) 
        
        # --- 5. Calcular el Delta (dΨ/dt = A * Ψ) usando einsum ---
        # A (bxyij):      [B, H, W, D, D] (b=batch, xy=coords, ij=matriz A)
        # x_cat (bjxy):   [B, D, H, W]    (b=batch, j=vector psi, xy=coords)
        # Ecuación: delta_psi[b, i, x, y] = sum_j ( A[b, x, y, i, j] * psi[b, j, x, y] )
        delta_psi = torch.einsum('bxyij,bjxy -> bixy', A, x_cat)
        
        # --- 6. Devolver el Delta ---
        # delta_psi ya está en el formato [B, C, H, W] que necesita el motor.
        return delta_psi