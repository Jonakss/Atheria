# src/qca_engine.py
import torch
import torch.nn as nn
from .config import DEVICE, D_STATE # ¡Importa la nueva config!

# ------------------------------------------------------------------------------
# 1.1: QCA_State Class (Revisada para ser Unitaria)
# ------------------------------------------------------------------------------
class QCA_State:
    def __init__(self, size, d_vector):
        self.size = size
        self.d_vector = d_vector
        # ¡¡SIMPLIFICADO!! Un solo vector de estado real.
        # Forma: [Batch, Altura, Ancho, Dimensiones_Vector]
        self.psi = torch.zeros(1, size, size, d_vector, device=DEVICE)
        # NUEVO: Estado de voltaje para la SNN
        self.v_mem = torch.zeros(1, size, size, d_vector, device=DEVICE)

    def _reset_state_random(self):
        """Inicializa el estado con ruido de baja amplitud."""
        noise = (torch.rand(1, self.size, self.size, self.d_vector, device=DEVICE) * 2 - 1) * 1e-2
        self.psi.data = noise
        self.v_mem.data.zero_() # Reiniciar voltaje
        # ¡No se necesita normalización!
        
    def _reset_state_complex_noise(self):
        """Inicializa el estado con ruido estructurado."""
        # (Esto es solo un ejemplo, puedes mejorarlo)
        noise = (torch.rand(1, self.size, self.size, self.d_vector, device=DEVICE) * 2 - 1) * 1e-3
        y_coords, x_coords = torch.meshgrid(torch.linspace(-1, 1, self.size, device=DEVICE),
                                            torch.linspace(-1, 1, self.size, device=DEVICE),
                                            indexing='ij')
        
        if self.d_vector > 0:
            pattern1 = torch.sin(x_coords * 10) * 0.1
            noise[0, :, :, 0] += pattern1
        if self.d_vector > 1:
            pattern2 = torch.cos(y_coords * 12) * 0.1
            noise[0, :, :, 1] += pattern2
            
        self.psi.data = noise
        self.v_mem.data.zero_() # Reiniciar voltaje

    def get_cat_input(self):
        """Prepara el estado para la U-Net (B, C, H, W)."""
        # [B, H, W, C] -> [B, C, H, W]
        return self.psi.permute(0, 3, 1, 2)
        
    def normalize_(self):
        """Normalizes the entire grid state to have a constant total norm."""
        # Calculate the total norm squared of the entire grid
        total_norm_sq = torch.sum(self.psi.pow(2))
        
        # Avoid division by zero
        total_norm = torch.sqrt(total_norm_sq) + 1e-9
        
        # Normalize the entire tensor in-place
        self.psi.data /= total_norm

# ------------------------------------------------------------------------------
# 1.3: Aetheria_Motor Class (Revisada para ser Unitaria)
# ------------------------------------------------------------------------------
class Aetheria_Motor:
    def __init__(self, size, d_vector, operator_model: nn.Module):
        self.size = size
        self.operator = operator_model.to(DEVICE)
        if torch.cuda.device_count() > 1:
            self.operator = nn.DataParallel(self.operator)
        
        # El motor usa el nuevo estado
        self.state = QCA_State(size, d_vector)

    def evolve_step(self):
        """Evoluciona el estado usando la Ley M (unitaria o SNN)."""
        with torch.no_grad():
            # Importar SNN_UNET aquí para evitar dependencia circular
            from .models.snn_unet import SNN_UNET

            # 1. Preparar el estado (B, C, H, W)
            x_cat_total = self.state.get_cat_input()

            # 2. Comprobar si el operador es una SNN
            if isinstance(self.operator, SNN_UNET):
                # La Ley M (SNN) toma el estado y el voltaje, y devuelve picos y nuevo voltaje
                spikes, new_v_mem = self.operator(x_cat_total, self.state.v_mem.permute(0, 3, 1, 2))
                
                # Los picos actúan como el delta
                delta_psi = spikes
                
                # Guardar el nuevo voltaje
                self.state.v_mem.data = new_v_mem.permute(0, 2, 3, 1)
            else:
                # La Ley M (U-Net) predice el Delta (la derivada)
                delta_psi = self.operator(x_cat_total) # Salida: [B, C, H, W]
            
            # 3. Aplicar el Método de Euler
            new_psi = self.state.psi + delta_psi.permute(0, 2, 3, 1)

            # 4. Guardar el estado t+1
            self.state.psi.data = new_psi