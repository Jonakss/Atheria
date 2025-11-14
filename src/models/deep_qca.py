# src/models/deep_qca.py
import torch
import torch.nn as nn



class DeepQCA(nn.Module):
    """
    La Ley M "Profunda" original. Es un MLP 1x1 aplicado a los vecinos.
    """
    def __init__(self, d_state, hidden_channels, **kwargs):
        super().__init__()
        self.d_state = d_state
        
        # Convolución de Vecindad (No entrenable)
        self.conv_neighbors = nn.Conv2d(2*d_state, 2*d_state*8, kernel_size=3,
                                        padding=1, groups=2*d_state, bias=False)
        weights = torch.ones(2*d_state*8, 1, 3, 3)
        weights[:, 0, 1, 1] = 0.0
        self.conv_neighbors.weight.data = weights
        self.conv_neighbors.weight.requires_grad = False

        # MLP 1x1 (Entrenable)
        self.processing_net = nn.Sequential(
            nn.Conv2d(2 * d_state * 8, hidden_channels, kernel_size=1),
            nn.ELU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ELU(inplace=False),
            nn.Conv2d(hidden_channels, 2 * d_state, kernel_size=1) # Salida directa al delta
        )
        nn.init.normal_(self.processing_net[-1].weight, mean=0.0, std=1e-5)

    def forward(self, x_cat):
        """Aplica la Ley M local."""
        self.conv_neighbors.weight.data = self.conv_neighbors.weight.data.to(x_cat.device)
        x_neighbors = self.conv_neighbors(x_cat)
        
        # El MLP ahora predice directamente el delta agregado de los vecinos
        delta_psi_from_neighbors = self.processing_net(x_neighbors)
        
        # Agregamos los deltas de los vecinos (es una suma ponderada por la red)
        # La red aprende a mapear el estado de los vecinos a un delta para la célula central
        return delta_psi_from_neighbors