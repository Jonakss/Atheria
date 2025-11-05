# src/qca_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importar la configuración del dispositivo (¡importación relativa!)
from .config import DEVICE

# ------------------------------------------------------------------------------
# 1.1: QCA_State Class (El "Lienzo" del Universo)
# ------------------------------------------------------------------------------
class QCA_State:
    def __init__(self, size, d_state):
        self.size = size
        self.d_state = d_state
        self.x_real = torch.zeros(1, size, size, d_state, device=DEVICE)
        self.x_imag = torch.zeros(1, size, size, d_state, device=DEVICE)

    def _reset_state_random(self):
        """Initializes the state with low-amplitude noise and normalizes it."""
        noise_r = (torch.rand(1, self.size, self.size, self.d_state, device=DEVICE) * 2 - 1) * 1e-2
        noise_i = (torch.rand(1, self.size, self.size, self.d_state, device=DEVICE) * 2 - 1) * 1e-2
        self.x_real.data = noise_r
        self.x_imag.data = noise_i
        self.normalize_state()

    def _reset_state_seeded(self):
        """Initializes the state with a vacuum and a 'seed' of activity in the center."""
        self.x_real.data.fill_(0)
        self.x_imag.data.fill_(0)
        center_x, center_y = self.size // 2, self.size // 2
        seed_size = max(1, self.size // 64)
        for dx in range(-seed_size, seed_size + 1):
            for dy in range(-seed_size, seed_size + 1):
                if 0 <= center_x + dx < self.size and 0 <= center_y + dy < self.size:
                        if self.d_state > 3: # Ensure enough channels exist
                            self.x_real[0, center_y + dy, center_x + dx, 0] = 0.5
                            self.x_imag[0, center_y + dy, center_x + dx, 1] = 0.5
                            self.x_real[0, center_y + dy, center_x + dx, 2] = -0.5
                            self.x_imag[0, center_y + dy, center_x + dx, 3] = -0.5
        self.normalize_state()

    def _reset_state_complex_noise(self):
        """Initializes the state with a structured complex noise pattern."""
        y_coords, x_coords = torch.meshgrid(torch.linspace(-1, 1, self.size, device=DEVICE),
                                            torch.linspace(-1, 1, self.size, device=DEVICE),
                                            indexing='ij')
        radial_dist = torch.sqrt(x_coords**2 + y_coords**2)
        angle = torch.atan2(y_coords, x_coords)
        pattern1_r = torch.sin(x_coords * 10 + angle * 5) * 0.1
        pattern1_i = torch.cos(y_coords * 12 + angle * 6) * 0.1
        pattern2_r = torch.sin(radial_dist * 15 + x_coords * 8) * 0.05
        pattern2_i = torch.cos(radial_dist * 18 + y_coords * 9) * 0.05
        noise_r = (torch.rand(self.size, self.size, self.d_state, device=DEVICE) * 2 - 1) * 1e-3
        noise_i = (torch.rand(self.size, self.size, self.d_state, device=DEVICE) * 2 - 1) * 1e-3
        if self.d_state > 0: noise_r[:, :, 0] += pattern1_r
        if self.d_state > 1: noise_i[:, :, 1] += pattern1_i
        if self.d_state > 2: noise_r[:, :, 2] += pattern2_r
        if self.d_state > 3: noise_i[:, :, 3] += pattern2_i
        self.x_real.data = noise_r.unsqueeze(0).to(DEVICE)
        self.x_imag.data = noise_i.unsqueeze(0).to(DEVICE)
        self.normalize_state()

    def normalize_state(self):
        """Normalizes the state vector in each cell to conserve probability."""
        prob_sq = self.x_real.pow(2) + self.x_imag.pow(2)
        norm = torch.sqrt(prob_sq.sum(dim=-1, keepdim=True) + 1e-8)
        self.x_real.data = self.x_real.data / norm
        self.x_imag.data = self.x_imag.data / norm

    def get_cat(self):
        """Concatenates real/imag tensors into (B, C, H, W) format for CNNs."""
        x_real_c = self.x_real.permute(0, 3, 1, 2)
        x_imag_c = self.x_imag.permute(0, 3, 1, 2)
        return torch.cat([x_real_c, x_imag_c], dim=1)

# ------------------------------------------------------------------------------
# 1.3: Aetheria_Motor Class (El "Motor de Evolución")
# ------------------------------------------------------------------------------
class Aetheria_Motor:
    def __init__(self, size, d_state, operator_model: nn.Module):
        """
        Inicializa el motor.
        
        Args:
            size (int): Tamaño de la cuadrícula (ej. 256).
            d_state (int): Dimensión del estado (ej. 21).
            operator_model (nn.Module): La "Ley M" (MLP, U-Net, etc.)
                                         que se usará para la evolución.
        """
        self.size = size
        self.d_state = d_state
        
        self.operator = operator_model.to(DEVICE)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Usando {torch.cuda.device_count()} GPUs para el operador.")
            self.operator = nn.DataParallel(self.operator)

        self.state = QCA_State(size, d_state)

    def evolve_step(self):
        """Evolves the QCA state one time step using the provided operator model."""
        with torch.no_grad():
            prev_state = self.state
            x_cat = prev_state.get_cat() 

            if isinstance(self.operator, nn.DataParallel):
                x_cat = x_cat.to(self.operator.device_ids[0])
            else:
                 x_cat = x_cat.to(DEVICE)

            # --- 1. Aplicar la Ley M (El "Cerebro") ---
            delta_real, delta_imag = self.operator(x_cat)

            # --- 2. Manejar los Biases (si existen) ---
            bias_real = 0.0
            bias_imag = 0.0
            
            op_to_check = self.operator.module if isinstance(self.operator, nn.DataParallel) else self.operator
            
            if hasattr(op_to_check, 'M_bias_real'):
                bias_real = op_to_check.M_bias_real.to(delta_real.device)
            if hasattr(op_to_check, 'M_bias_imag'):
                bias_imag = op_to_check.M_bias_imag.to(delta_imag.device)

            # --- 3. Actualizar el Estado (La Física) ---
            new_real = prev_state.x_real.squeeze(0) + delta_real + bias_real
            new_imag = prev_state.x_imag.squeeze(0) + delta_imag + bias_imag

            # Normalización
            prob_sq = new_real.pow(2) + new_imag.pow(2)
            norm = torch.sqrt(prob_sq.sum(dim=-1, keepdim=True) + 1e-8)
            next_real = new_real / norm
            next_imag = new_imag / norm

            self.state.x_real.data = next_real.unsqueeze(0)
            self.state.x_imag.data = next_imag.unsqueeze(0)