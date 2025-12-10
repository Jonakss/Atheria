
import torch
import torch.nn as nn

class LagrangianNetwork(nn.Module):
    """
    Lagrangian Neural Network (LNN) para el proyecto Aetheria.
    
    Aprende el Lagrangiano L(q, q_dot) directamente.
    Asume que el sistema es un campo discreto en un grid (Cellular Automata),
    por lo que L es la suma de densidades Lagrangianas locales.
    
    Architecture:
        - Input: Concatenación de q (state) y v (velocity) -> [B, 2*C, H, W]
        - Output: Scalar L (sum of all cells) or Density map [B, 1, H, W]
        
    Para permitir la inversión eficiente del Hessiano d2L/dv2, asumimos
    que la dependencia en 'v' es local (pixel-wise), similar a la energía cinética standard.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = 2 * state_dim  # q y q_dot
        
        # Usamos Convoluciones 1x1 para simular un MLP actuando en cada celda independientemente.
        # Esto asegura que d2L/dvi dvj es cero si i,j son de celdas distintas,
        # lo que hace que el Hessiano sea bloque-diagonal (invertible por pixel).
        # Para interacciones espaciales, deberíamos agregar convs 3x3 en la rama de 'q',
        # pero para el término cinético (v) es mejor mantenerlo local.
        
        self.net = nn.Sequential(
            # Capa 1: Mezcla q y v localmente
            nn.Conv2d(self.input_dim, hidden_dim, kernel_size=1),
            nn.Softplus(), # Softplus es suave (C-infinito), mejor que ReLU para derivadas segundas
            
            # Capa 2: Procesamiento
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Softplus(),
            
            # Capa 3: Salida escalar (Densidad Lagrangiana)
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        
        # Inicialización conservadora cerca de cero
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, q, v):
        """
        Args:
            q: Estado [B, C, H, W]
            v: Velocidad [B, C, H, W]
            
        Returns:
            L_density: [B, 1, H, W] Densidad Lagrangiana por celda
        """
        # Concatenar a lo largo del canal
        x = torch.cat([q, v], dim=1)
        return self.net(x)

    def compute_lagrangian(self, q, v):
        """Retorna la Acción total (suma escalar) para un batch."""
        density = self.forward(q, v)
        return density.sum(dim=[1, 2, 3]) # Suma sobre C, H, W -> [B]
