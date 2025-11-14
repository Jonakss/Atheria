# src/qca_engine.py
import torch
import torch.nn as nn
import os
from . import config as cfg

class QuantumState:
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

    def _reset_state_random(self):
        self.psi = self._initialize_state(mode='random')

    def load_state(self, filepath):
        try:
            state_dict = torch.load(filepath, map_location=self.device)
            self.psi = state_dict['psi'].to(self.device)
            print(f"Estado cargado desde {filepath}")
        except Exception as e:
            print(f"Error al cargar el estado: {e}. Reiniciando a un estado aleatorio.")
            self._reset_state_random()

    def save_state(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({'psi': self.psi.cpu()}, filepath)


class Aetheria_Motor:
    def __init__(self, model_operator: nn.Module, grid_size: int, d_state: int, device):
        self.device = device
        # --- ¡¡CORRECCIÓN CLAVE!! Asegurar que los tamaños son enteros ---
        if grid_size is None or d_state is None:
            raise ValueError(f"grid_size y d_state no pueden ser None. Recibido: grid_size={grid_size}, d_state={d_state}")
        self.grid_size = int(grid_size)
        self.d_state = int(d_state)
        
        self.operator = model_operator.to(self.device)
        self.state = QuantumState(self.grid_size, self.d_state, device)
        self.is_compiled = False
        self.cfg = cfg

    def get_initial_state(self, batch_size: int):
        """
        Genera un estado inicial aleatorio y normalizado para el entrenamiento.
        """
        real = torch.randn(batch_size, self.grid_size, self.grid_size, self.d_state, device=self.device)
        imag = torch.randn(batch_size, self.grid_size, self.grid_size, self.d_state, device=self.device)
        psi_complex = torch.complex(real, imag)
        norm = torch.sqrt(torch.sum(psi_complex.abs().pow(2), dim=-1, keepdim=True))
        return psi_complex / norm

    def propagate(self, psi_inicial, num_steps):
        """
        Propaga un estado inicial a lo largo de un número de pasos.
        """
        psi_actual = psi_inicial
        for _ in range(num_steps):
            # Calcular x_cat a partir del estado actual
            x_cat_real = psi_actual.real.permute(0, 3, 1, 2)
            x_cat_imag = psi_actual.imag.permute(0, 3, 1, 2)
            x_cat_total = torch.cat([x_cat_real, x_cat_imag], dim=1)
            
            # Usar evolve_step para obtener el siguiente estado
            psi_actual = self.evolve_step(is_training=True, x_cat=x_cat_total)
        return psi_actual



    def compile_model(self):
        if not self.is_compiled:
            try:
                print("Aplicando torch.compile() al modelo...")
                self.operator = torch.compile(self.operator, mode="reduce-overhead")
                self.is_compiled = True
                print("¡torch.compile() aplicado exitosamente!")
            except Exception as e:
                print(f"torch.compile() falló: {e}. El modelo se ejecutará sin compilar.")

    def evolve_step(self, is_training=False, x_cat=None):
        """
        Ejecuta un paso de evolución. La física ahora es condicional.
        """
        if is_training and x_cat is not None:
            # En modo entrenamiento, el delta se calcula a partir del x_cat proporcionado
            delta_psi_unitario_complex = self.operator(x_cat)
        else:
            # En modo simulación, calculamos el x_cat internamente
            with torch.no_grad():
                x_cat_real = self.state.psi.real.permute(0, 3, 1, 2)
                x_cat_imag = self.state.psi.imag.permute(0, 3, 1, 2)
                x_cat_total = torch.cat([x_cat_real, x_cat_imag], dim=1)
                delta_psi_unitario_complex = self.operator(x_cat_total)

        delta_real, delta_imag = torch.chunk(delta_psi_unitario_complex, 2, dim=1)
        delta_psi_unitario = torch.complex(delta_real, delta_imag).permute(0, 2, 3, 1)

        if hasattr(self.cfg, 'GAMMA_DECAY') and self.cfg.GAMMA_DECAY > 0:
            delta_psi_decay = -self.cfg.GAMMA_DECAY * self.state.psi
            delta_psi_total = delta_psi_unitario + delta_psi_decay
            new_psi = self.state.psi + delta_psi_total
            if not is_training: self.state.psi = new_psi
        else:
            new_psi = self.state.psi + delta_psi_unitario
            norm = torch.sqrt(torch.sum(new_psi.abs().pow(2), dim=-1, keepdim=True))
            new_psi = new_psi / norm
            if not is_training: self.state.psi = new_psi
            
        return new_psi