# src/models/snn_unet.py
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNNUNet(nn.Module):
    """
    Una U-Net de Picos (Spiking U-Net) robusta con gestión manual de estado.
    """
    def __init__(self, d_state, hidden_channels, alpha=0.9, beta=0.85, **kwargs):
        super().__init__()
        
        in_channels = 2 * d_state
        out_channels = 2 * d_state

        self.beta = beta
        self.grad = surrogate.atan()

        # --- ¡¡CORRECCIÓN CLAVE!! Eliminar init_hidden=True ---
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.grad)
        
        self.conv_bottom = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.lif_bottom = snn.Leaky(beta=self.beta, spike_grad=self.grad)

        self.upconv = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.grad)

        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.lif_out = snn.Leaky(beta=self.beta, spike_grad=self.grad, output=True)

    def forward(self, x):
        """
        Forward pass con gestión manual del estado de la membrana.
        """
        # --- ¡¡CORRECCIÓN CLAVE!! Reiniciar estados manualmente ---
        mem1 = self.lif1.init_leaky()
        mem_bottom = self.lif_bottom.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        cur1 = self.conv1(x)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur_bottom = self.conv_bottom(spk1)
        spk_bottom, mem_bottom = self.lif_bottom(cur_bottom, mem_bottom)

        up = self.upconv(spk_bottom)
        cat = torch.cat([up, spk1], dim=1)
        cur2 = self.conv2(cat)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur_out = self.conv_out(spk2)
        spk_out, mem_out = self.lif_out(cur_out, mem_out)

        return spk_out
