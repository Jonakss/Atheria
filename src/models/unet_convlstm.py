# src/models/unet_convlstm.py
"""
U-Net con ConvLSTM para dar memoria temporal a la Ley M.
Esta arquitectura permite que el universo "recuerde" eventos pasados.
"""
import torch
import torch.nn as nn
import logging


class ConvBlock(nn.Module):
    """(Conv => GroupNorm => ELU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        num_groups_mid = max(1, mid_channels // 8 if mid_channels > 8 else 1)
        if mid_channels % num_groups_mid != 0: 
            num_groups_mid = 1
             
        num_groups_out = max(1, out_channels // 8 if out_channels > 8 else 1)
        if out_channels % num_groups_out != 0: 
            num_groups_out = 1
             
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


class UNetConvLSTM(nn.Module):
    """
    U-Net con ConvLSTM en el bottleneck para dar memoria temporal.
    
    Esta arquitectura permite que la Ley M tenga "memoria" del pasado,
    lo que habilita comportamientos temporales complejos como:
    - Osciladores (relojes internos)
    - Causalidad de largo alcance
    - "ADN" emergente (genomas locales)
    """
    _compiles = False  # Desactivar torch.compile para este modelo (ConvLSTM puede tener problemas)
    
    def __init__(self, d_state, hidden_channels, d_memory=None, **kwargs):
        super().__init__()
        self.d_state = d_state
        self.d_memory = d_memory if d_memory is not None else hidden_channels // 2
        
        base_c = hidden_channels
        in_c = 2 * d_state  # real + imag
        out_c = 2 * d_state  # delta real + delta imag
        
        # Capa de entrada
        self.inc = ConvBlock(in_c, base_c)
        
        # Encoder (downsampling)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_c * 2, base_c * 4))
        
        # Bottleneck con ConvLSTM (aquí está la memoria)
        # Implementamos ConvLSTM manualmente ya que PyTorch no tiene ConvLSTM2d nativo
        # ConvLSTM combina convoluciones 2D con LSTM para mantener memoria espacial y temporal
        self.hidden_channels = base_c * 4
        
        # Gates del ConvLSTM: forget, input, output, candidate
        self.conv_gates = nn.Conv2d(
            base_c * 4 + self.hidden_channels,  # input + hidden
            4 * self.hidden_channels,  # 4 gates
            kernel_size=3,
            padding=1
        )
        
        # Bottleneck conv después de ConvLSTM
        self.bot = ConvBlock(base_c * 4, base_c * 8)
        
        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(base_c * 6, base_c * 4)  # 6 = 4 (up) + 2 (skip)
        
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(base_c * 3, base_c)  # 3 = 2 (up) + 1 (skip)
        
        # Capa de salida
        self.outc = nn.Conv2d(base_c, out_c, kernel_size=1)
        nn.init.normal_(self.outc.weight, mean=0.0, std=1e-5)
        
        logging.info(f"UNetConvLSTM inicializado: d_state={d_state}, hidden_channels={base_c}, d_memory={self.d_memory}")
    
    def forward(self, x_cat, h_t=None, c_t=None):
        """
        Forward pass con memoria temporal.
        
        Args:
            x_cat: Tensor de entrada [batch, 2*d_state, H, W] (real e imag concatenados)
            h_t: Estado oculto de ConvLSTM [1, batch, base_c*4, H_bot, W_bot] o None
            c_t: Estado de celda de ConvLSTM [1, batch, base_c*4, H_bot, W_bot] o None
        
        Returns:
            delta_psi: Delta del estado [batch, 2*d_state, H, W]
            h_next: Nuevo estado oculto [1, batch, base_c*4, H_bot, W_bot]
            c_next: Nuevo estado de celda [1, batch, base_c*4, H_bot, W_bot]
        """
        # Encoder
        x1 = self.inc(x_cat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)  # [batch, base_c*4, H_bot, W_bot]
        
        # ConvLSTM: procesar x3 con memoria h_t y c_t
        batch_size, channels, H, W = x3.shape
        
        # Inicializar estados si no existen
        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_channels, H, W, 
                            device=x3.device, dtype=x3.dtype)
        else:
            # Si h_t viene en formato [1, batch, channels, H, W], ajustar
            if h_t.dim() == 5:
                h_t = h_t.squeeze(0)  # [batch, channels, H, W]
        
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_channels, H, W, 
                            device=x3.device, dtype=x3.dtype)
        else:
            # Si c_t viene en formato [1, batch, channels, H, W], ajustar
            if c_t.dim() == 5:
                c_t = c_t.squeeze(0)  # [batch, channels, H, W]
        
        # Concatenar input y hidden state para los gates
        combined = torch.cat([x3, h_t], dim=1)  # [batch, base_c*4 + hidden_channels, H, W]
        
        # Calcular gates
        gates = self.conv_gates(combined)  # [batch, 4*hidden_channels, H, W]
        
        # Dividir en 4 gates
        f_gate, i_gate, o_gate, g_gate = torch.chunk(gates, 4, dim=1)
        
        # Aplicar activaciones
        f_gate = torch.sigmoid(f_gate)  # Forget gate
        i_gate = torch.sigmoid(i_gate)  # Input gate
        o_gate = torch.sigmoid(o_gate)  # Output gate
        g_gate = torch.tanh(g_gate)     # Candidate values
        
        # Actualizar estados de celda y hidden
        c_next = f_gate * c_t + i_gate * g_gate  # Nueva celda
        h_next = o_gate * torch.tanh(c_next)      # Nuevo hidden
        
        # El output del ConvLSTM es h_next
        x3_mem = h_next  # [batch, hidden_channels, H_bot, W_bot]
        
        # Asegurar que h_next y c_next tengan el formato esperado [1, batch, channels, H, W]
        h_next = h_next.unsqueeze(0)  # [1, batch, channels, H, W]
        c_next = c_next.unsqueeze(0)  # [1, batch, channels, H, W]
        
        # Bottleneck conv
        b = self.bot(x3_mem)
        
        # Decoder
        u1 = self.up1(b)
        s1 = torch.cat([u1, x2], dim=1)
        c1 = self.conv_up1(s1)
        
        u2 = self.up2(c1)
        s2 = torch.cat([u2, x1], dim=1)
        c2 = self.conv_up2(s2)
        
        # Salida
        delta_psi = self.outc(c2)
        
        return delta_psi, h_next, c_next

