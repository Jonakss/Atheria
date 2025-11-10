# src/models/mlp.py
# Contiene la Ley M original (basada en MLP 1x1, "míope").
import torch
import torch.nn as nn

from ..config import DEVICE
from . import register_model

@register_model('MLP')
class QCA_Operator_MLP(nn.Module):
    """
    La Ley M "Profunda" original (QCA_Operator_Deep).
    Es un MLP 1x1 aplicado a los vecinos.
    Es agnóstico al tamaño, pero solo tiene visión local.
    """
    def __init__(self, d_state, hidden_channels):
        super().__init__()
        self.d_state = d_state
        
        # 1. Convolución de Vecindad (No entrenable)
        self.conv_neighbors = nn.Conv2d(2*d_state, 2*d_state*8, kernel_size=3,
                                        padding=1, groups=2*d_state, bias=False)
        weights = torch.ones(2*d_state*8, 1, 3, 3)
        weights[:, 0, 1, 1] = 0.0 # Mascara central
        self.conv_neighbors.weight.data = weights
        self.conv_neighbors.weight.requires_grad = False

        # 2. El MLP 1x1 (Entrenable)
        self.processing_net = nn.Sequential(
            nn.Conv2d(2 * d_state * 8, hidden_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, 8 * d_state, kernel_size=1, bias=False)
        )

        # 3. Biases (Entrenables) - ¡Importante! El motor los busca.
        self.M_bias_real = nn.Parameter(torch.zeros(d_state))
        self.M_bias_imag = nn.Parameter(torch.zeros(d_state))

    def forward(self, x_cat):
        """Aplica la Ley M local."""
        # Mover pesos a la GPU correcta (necesario si el modelo se mueve después de init)
        self.conv_neighbors.weight.data = self.conv_neighbors.weight.data.to(x_cat.device)
        
        x_neighbors = self.conv_neighbors(x_cat)
        F_int = self.processing_net(x_neighbors) # [B, 8*d_state, H, W]

        # Reformatear a la salida esperada
        F_int = F_int.squeeze(0).permute(1, 2, 0) # [H, W, 8*d_state]
        H, W, C = F_int.shape
        D4 = 4 * self.d_state

        F_int_real_raw = F_int[..., :D4]
        F_int_imag_raw = F_int[..., D4:]
        
        F_int_real = F_int_real_raw.reshape(H, W, 4, self.d_state).mean(dim=2) * 0.1
        F_int_imag = F_int_imag_raw.reshape(H, W, 4, self.d_state).mean(dim=2) * 0.1

        return F_int_real, F_int_imag