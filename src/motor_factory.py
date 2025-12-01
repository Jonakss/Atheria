import logging
import torch.nn as nn
from .engines.qca_engine import Aetheria_Motor
from .engines.qca_engine_polar import PolarEngine
from .engines.compute_backend import LocalBackend, MockQuantumBackend, ComputeBackend

def get_motor(config, model: nn.Module, device):
    """
    Factory function to create the appropriate physics engine based on configuration.
    
    Args:
        config: Configuration object or dict containing ENGINE_TYPE
        model: The neural network model (operator)
        device: The device to run on (cpu/cuda) - DEPRECATED in favor of backend, but kept for compat
        
    Returns:
        An instance of the selected motor class
    """
    # Extract engine type from config
    engine_type = 'CARTESIAN'
    if hasattr(config, 'ENGINE_TYPE'):
        engine_type = config.ENGINE_TYPE
    elif isinstance(config, dict) and 'ENGINE_TYPE' in config:
        engine_type = config['ENGINE_TYPE']
        
    # Determine Backend
    # TODO: In the future, read BACKEND_TYPE from config. Default to LocalBackend.
    backend_type = getattr(config, 'BACKEND_TYPE', 'LOCAL')
    
    backend: ComputeBackend
    if backend_type == 'QUANTUM_MOCK':
        backend = MockQuantumBackend()
    else:
        # Default to LocalBackend using the passed device
        backend = LocalBackend(device)
        
    logging.info(f"🏭 Motor Factory: Requesting engine type '{engine_type}' on backend '{backend.__class__.__name__}'")
    
    # Extract grid size and d_state from config or model
    # Try to get from config first
    grid_size = 128
    if hasattr(config, 'GRID_SIZE'):
        grid_size = config.GRID_SIZE
    elif isinstance(config, dict) and 'GRID_SIZE' in config:
        grid_size = config['GRID_SIZE']
        
    d_state = 16
    if hasattr(config, 'D_STATE'):
        d_state = config.D_STATE
    elif isinstance(config, dict) and 'D_STATE' in config:
        d_state = config['D_STATE']
        
    # Validation: Check for compatibility
    # TODO: Add more robust validation checking model architecture metadata if available
    
    if engine_type == 'POLAR':
        logging.info("🌀 Initializing Polar Engine (Rotational/Stability optimized)")
        # Check if model is compatible with Polar engine if possible
        # For now, we assume the user knows what they are doing or the model is generic enough
        return PolarEngine(model, grid_size)
        
    elif engine_type == 'QUANTUM':
        logging.info("⚛️ Initializing Quantum Hybrid Engine")
        # Placeholder for Quantum Engine
        # For now fallback to Cartesian or raise NotImplementedError
        logging.warning("Quantum Engine not fully implemented, falling back to Cartesian with Quantum flags")
        # Pass backend to engine (Engine needs update to accept it)
        return Aetheria_Motor(model, grid_size, d_state, backend.get_device(), cfg=config)
        
    else: # CARTESIAN or default
        if engine_type != 'CARTESIAN':
            logging.warning(f"Unknown engine type '{engine_type}', defaulting to CARTESIAN")
            
        logging.info("🌊 Initializing Standard Cartesian Engine")
        return Aetheria_Motor(model, grid_size, d_state, backend.get_device(), cfg=config)
