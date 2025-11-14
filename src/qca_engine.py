# src/qca_engine.py
import torch
import torch.nn as nn
import os
import logging
from . import config as cfg

class QuantumState:
    # ... (Clase sin cambios, la incluyo para completitud)
    def __init__(self, grid_size, d_state, device):
        self.grid_size = grid_size
        self.d_state = d_state
        self.device = device
        self.psi = self._initialize_state()
    def _initialize_state(self, mode='complex_noise', complex_noise_strength=0.1):
        if mode == 'random':
            real = torch.randn(1, self.grid_size, self.grid_size, self.d_state, device=self.device)
            imag = torch.randn(1, self.grid_size, self.grid_size, self.d_state, device=self.device)
            psi_complex = torch.complex(real, imag)
            norm = torch.sqrt(torch.sum(psi_complex.abs().pow(2), dim=-1, keepdim=True))
            return psi_complex / norm
        elif mode == 'complex_noise':
            noise = torch.randn(1, self.grid_size, self.grid_size, self.d_state, device=self.device) * complex_noise_strength
            real, imag = torch.cos(noise), torch.sin(noise)
            return torch.complex(real, imag)
        else:
            return torch.zeros(1, self.grid_size, self.grid_size, self.d_state, device=self.device, dtype=torch.complex64)
    def _reset_state_random(self): self.psi = self._initialize_state(mode='random')
    def load_state(self, filepath):
        try:
            state_dict = torch.load(filepath, map_location=self.device)
            self.psi = state_dict['psi'].to(self.device)
        except Exception: self._reset_state_random()
    def save_state(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({'psi': self.psi.cpu()}, filepath)

class Aetheria_Motor:
    def __init__(self, model_operator: nn.Module, grid_size: int, d_state: int, device):
        self.device = device
        self.grid_size = int(grid_size)
        self.d_state = int(d_state)
        self.operator = model_operator.to(self.device)
        self.state = QuantumState(self.grid_size, self.d_state, device)
        self.is_compiled = False
        self.cfg = cfg

    def evolve_internal_state(self):
        if self.state.psi is None: return
        with torch.no_grad():
            self.state.psi = self._evolve_logic(self.state.psi)

    def evolve_step(self, current_psi):
        with torch.set_grad_enabled(True):
            return self._evolve_logic(current_psi)

    def _evolve_logic(self, psi_in):
        x_cat_real = psi_in.real.permute(0, 3, 1, 2)
        x_cat_imag = psi_in.imag.permute(0, 3, 1, 2)
        x_cat_total = torch.cat([x_cat_real, x_cat_imag], dim=1)
        delta_psi_unitario_complex = self.operator(x_cat_total)

        delta_real, delta_imag = torch.chunk(delta_psi_unitario_complex, 2, dim=1)
        delta_psi_unitario = torch.complex(delta_real, delta_imag).permute(0, 2, 3, 1)
        
        model_name = self.operator.__class__.__name__
        if model_name == 'UNetUnitary':
            A_times_psi_real = delta_psi_unitario.real * psi_in.real - delta_psi_unitario.imag * psi_in.imag
            A_times_psi_imag = delta_psi_unitario.real * psi_in.imag + delta_psi_unitario.imag * psi_in.real
            delta_psi_unitario = torch.complex(A_times_psi_real, A_times_psi_imag)

        if hasattr(self.cfg, 'GAMMA_DECAY') and self.cfg.GAMMA_DECAY > 0:
            delta_psi_decay = -self.cfg.GAMMA_DECAY * psi_in
            delta_psi_total = delta_psi_unitario + delta_psi_decay
            new_psi = psi_in + delta_psi_total
        else:
            new_psi = psi_in + delta_psi_unitario
        
        norm = torch.sqrt(torch.sum(new_psi.abs().pow(2), dim=-1, keepdim=True))
        # Evitar división por cero si la norma es muy pequeña
        new_psi = new_psi / (norm + 1e-9)
        
        return new_psi

    def propagate(self, psi_inicial, num_steps):
        psi_history = []
        psi_actual = psi_inicial
        for _ in range(num_steps):
            psi_actual = self.evolve_step(psi_actual)
            psi_history.append(psi_actual)
        return psi_history, psi_actual
    
    def get_initial_state(self, batch_size: int):
        real = torch.randn(batch_size, self.grid_size, self.grid_size, self.d_state, device=self.device)
        imag = torch.randn(batch_size, self.grid_size, self.grid_size, self.d_state, device=self.device)
        psi_complex = torch.complex(real, imag)
        norm = torch.sqrt(torch.sum(psi_complex.abs().pow(2), dim=-1, keepdim=True))
        return psi_complex / (norm + 1e-9)

    def laplacian_2d_psi(self, psi):
        psi_permuted = psi.permute(0, 3, 1, 2)
        d_state = psi_permuted.shape[1]
        kernel_base = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                   dtype=torch.float32, device=self.device).reshape(1, 1, 3, 3)
        kernel = kernel_base.repeat(d_state, 1, 1, 1)
        laplacian_real = nn.functional.conv2d(psi_permuted.real, kernel, padding=1, groups=d_state)
        laplacian_imag = nn.functional.conv2d(psi_permuted.imag, kernel, padding=1, groups=d_state)
        laplacian_complex = torch.complex(laplacian_real, laplacian_imag)
        return laplacian_complex.permute(0, 2, 3, 1)

    def compile_model(self):
        if not hasattr(self.operator, '_compiles') or self.operator._compiles:
            if not self.is_compiled:
                try:
                    logging.info("Aplicando torch.compile() al modelo...")
                    self.operator = torch.compile(self.operator, mode="reduce-overhead")
                    self.is_compiled = True
                    logging.info("¡torch.compile() aplicado exitosamente!")
                except Exception as e:
                    logging.warning(f"torch.compile() falló: {e}. El modelo se ejecutará sin compilar.")
        else:
            logging.info(f"torch.compile() omitido para el modelo {self.operator.__class__.__name__} según su configuración.")