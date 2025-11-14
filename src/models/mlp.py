# src/models/mlp.py
import torch
import torch.nn as nn



class MLP(nn.Module):
    """
    Un MLP simple que opera sobre cada celda de forma independiente (convolución 1x1).
    Es una versión simplificada para pruebas.
    """
    def __init__(self, d_state, hidden_channels, **kwargs):
        super().__init__()
        self.d_state = d_state
        in_channels = 2 * d_state
        out_channels = 2 * d_state

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ELU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ELU(inplace=False),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )
        # Inicializar la última capa con pesos pequeños para empezar con deltas pequeños
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-5)

    def forward(self, x_cat):
        """Aplica el MLP a cada celda."""
        return self.net(x_cat)
