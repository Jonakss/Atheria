# src/pipeline_viz.py
import torch
import numpy as np
from .qca_engine import Aetheria_Motor, QuantumState
from .visualization_tools import (
    get_density_map,
    get_channels_map,
    get_phase_map,
    get_change_map
)

class VisualizationPipeline:
    """
    Gestiona la visualización de una simulación en tiempo real.
    """
    def __init__(self, model, config):
        self.device = config.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        grid_size = config.get('GRID_SIZE_VIZ', 128)
        
        # Extraer d_state de MODEL_PARAMS, con un valor por defecto
        model_params = config.get('MODEL_PARAMS', {})
        d_state = model_params.get('d_state', 8)

        self.motor = Aetheria_Motor(model, grid_size, d_state, self.device)
        self.motor.compile_model()
        self.motor.state._reset_state_random()
        
        self.prev_psi = self.motor.state.psi.clone()
        self.step_count = 0

    def run_step(self):
        """
        Avanza un paso en la simulación y genera un diccionario de frames de visualización.
        """
        self.prev_psi = self.motor.state.psi.clone()
        
        # Evolucionar el estado usando el motor
        next_psi = self.motor.evolve_step()

        # --- ¡¡CORRECCIÓN CLAVE!! Usar las funciones correctas ---
        # Las funciones de visualization_tools ahora esperan tensores complejos.
        density_frame = get_density_map(next_psi)
        channel_frame = get_channels_map(next_psi)
        phase_frame = get_phase_map(next_psi)
        change_frame = get_change_map(next_psi, self.prev_psi)

        self.step_count += 1

        # Devolver los frames como arrays de numpy
        return {
            'density': np.array(density_frame).tolist(),
            'channels': np.array(channel_frame).tolist(),
            'phase': np.array(phase_frame).tolist(),
            'change': np.array(change_frame).tolist(),
        }