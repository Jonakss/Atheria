# src/models/snn_unet.py
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from . import register_model

@register_model("SNN_UNET")
class SNN_UNET(nn.Module):
    """
    Una U-Net de Picos (Spiking U-Net) simple para ser usada como un operador QCA.
    Esta es una implementación básica de un solo nivel.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=32, beta=0.9):
        super().__init__()
        
        # --- Hiperparámetros de la SNN ---
        self.beta = beta  # Factor de decaimiento del voltaje de la membrana
        self.grad = surrogate.atan() # Gradiente subrogado para el backward pass

        # --- Codificador ---
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True)
        
        # --- Capa Inferior (Bottleneck) ---
        self.conv_bottom = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.lif_bottom = snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True)

        # --- Decodificador ---
        self.upconv = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True)

        # --- Capa de Salida ---
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.lif_out = snn.Leaky(beta=self.beta, spike_grad=self.grad, init_hidden=True, output=True)

    def forward(self, x, mem_in):
        """
        Forward pass para un solo paso de tiempo.
        x: Estado de entrada (psi) - [B, C, H, W]
        mem_in: Estado de la membrana del paso anterior - [B, C, H, W]
        """
        # Reiniciar estados ocultos si es el primer paso (no implementado aquí, se asume un solo paso)
        
        # --- Codificador ---
        cur1 = self.conv1(x)
        spk1, mem1 = self.lif1(cur1)

        # --- Capa Inferior ---
        cur_bottom = self.conv_bottom(spk1)
        spk_bottom, mem_bottom = self.lif_bottom(cur_bottom)

        # --- Decodificador ---
        up = self.upconv(spk_bottom)
        # Skip connection
        cat = torch.cat([up, spk1], dim=1)
        cur2 = self.conv2(cat)
        spk2, mem2 = self.lif2(cur2)

        # --- Salida ---
        cur_out = self.conv_out(spk2)
        spk_out, mem_out = self.lif_out(cur_out, mem_in)

        # spk_out son los picos de salida (nuestro delta_psi)
        # mem_out es el nuevo estado de la membrana para el siguiente paso
        return spk_out, mem_out
