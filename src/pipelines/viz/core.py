"""Funciones core de cálculo de visualizaciones básicas."""
import torch
import numpy as np
import logging

from .utils import (
    apply_downsampling, 
    tensor_to_numpy, 
    synchronize_gpu, 
    get_inference_context,
    normalize_map_data
)
from .advanced import (
    calculate_poincare_coords,
    calculate_phase_attractor,
    calculate_flow_data,
    calculate_complex_3d_data,
    calculate_phase_hsv_data
)
from .phase_space import get_phase_space_data

# Threshold for quantum vacuum noise. Energy below this level is masked out.
VACUUM_THRESHOLD = 0.05

class CleanPolarWrapper:
    """Wrapper para estado polar con máscara de vacío aplicada."""
    def __init__(self, mag, phase, device=None):
        self.magnitude = mag
        self.phase = phase
        self.device = device

    def to_cartesian(self):
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        return real, imag

class CleanPolarWrapper:
    """Wrapper para estado polar con máscara de vacío aplicada."""
    def __init__(self, mag, phase, device=None):
        self.magnitude = mag
        self.phase = phase
        self.device = device

    def to_cartesian(self):
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        return real, imag

def get_visualization_data(psi, viz_type: str, delta_psi: torch.Tensor = None, motor=None, downsample_factor: int = 1):
    """
    Genera datos de visualización a partir del estado cuántico psi.
    
    Args:
        psi: Tensor complejo O objeto QuantumStatePolar
        viz_type: Tipo de visualización
        delta_psi: Tensor opcional
        motor: Instancia opcional de CartesianEngine
        downsample_factor: Factor de downsampling
    
    Returns:
        Dict con map_data, hist_data, etc.
    """
    # Validar entrada
    # Allow QuantumStatePolar object (duck typing)
    is_polar = hasattr(psi, 'magnitude') and hasattr(psi, 'phase')
    
    if not is_polar:
        validate_psi(psi)
        psi = normalize_psi_dimensions(psi)
    
    # DEBUG info
    if is_polar:
        # Assuming Polar state is (B, 1, H, W)
        mag = psi.magnitude
        logging.debug(f"🔍 Polar State stats: min_r={mag.min().item():.6f}, max_r={mag.max().item():.6f}")
    elif isinstance(psi, torch.Tensor):
        psi_min = psi.abs().min().item() if psi.numel() > 0 else 0.0
        psi_max = psi.abs().max().item() if psi.numel() > 0 else 0.0
        logging.debug(f"🔍 psi stats: min={psi_min:.6f}, max={psi_max:.6f}")

    # 2. MÁSCARA DE VACÍO (The Noise Filter)
    # Si la energía es menor al 5%, lo consideramos "Vacío Cuántico" y no lo dibujamos.
    # Esto limpia el gráfico 3D/Poincaré increíblemente.
    vacuum_threshold = 0.05

    if is_polar:
        mag = psi.magnitude
        mask = mag > vacuum_threshold
        # Create cleaned tensors
        clean_mag = mag * mask
        clean_pha = psi.phase * mask
        # Replace psi with cleaned wrapper
        psi = CleanPolarWrapper(clean_mag, clean_pha, getattr(psi, 'device', None))
    elif isinstance(psi, torch.Tensor):
        magnitude = torch.norm(psi, dim=-1)
        mask = magnitude > vacuum_threshold
        # Aplicar máscara (poner a cero lo que no es materia)
        psi = psi * mask.unsqueeze(-1)

    # Downsampling logic (simplified for Polar)
    # TODO: Implement downsampling for Polar if needed. For now skip.
    if not is_polar:
        psi = apply_downsampling(psi, downsample_factor)
    
    # Sincronizar GPU
    device = psi.device if (isinstance(psi, torch.Tensor) or hasattr(psi, 'device')) else torch.device('cpu')
    synchronize_gpu(device)
    
    # Calcular cantidades básicas
    density, phase, real_part, imag_part, energy = calculate_basic_quantities(psi)
    
    # Mover a CPU
    density = tensor_to_numpy(density, "density")
    phase = tensor_to_numpy(phase, "phase")
    real_part = tensor_to_numpy(real_part, "real_part")
    imag_part = tensor_to_numpy(imag_part, "imag_part")
    energy = tensor_to_numpy(energy, "energy")
    
    # Gradiente
    gradient_magnitude = calculate_gradient_magnitude(density)
    
    # Map Data
    map_data = select_map_data(
        viz_type, density, phase, real_part, imag_part, 
        gradient_magnitude, psi, motor
    )
    
    # Validations...
    if map_data is None or (hasattr(map_data, 'size') and map_data.size == 0):
        map_data = density
    
    if hasattr(map_data, 'shape') and len(map_data.shape) != 2:
         # Reshape logic if needed
         pass

    # Aplicar normalización absoluta para tipos de visualización físicos
    min_val, max_val = None, None
    if viz_type == 'density':
        # Densidad es energía |psi|^2. Mínimo físico 0, Máximo teórico 1.0
        min_val, max_val = 0.0, 1.0
    elif viz_type in ['phase', 'phase_hsv']:
        # Fase normalizada va de 0 a 1
        min_val, max_val = 0.0, 1.0

    map_data = normalize_map_data(map_data, min_val=min_val, max_val=max_val)
    
    result = {
        "map_data": map_data,
        "hist_data": {},
        "poincare_coords": [],
        "phase_attractor": None,
        "flow_data": None
    }
    
    # Advanced calculations based on viz_type
    if viz_type in ['poincare', 'poincare_3d']:
        result['poincare_coords'] = calculate_poincare_coords(psi)
    
    if viz_type == 'phase_attractor':
        result['phase_attractor'] = calculate_phase_attractor(psi)
        
    if viz_type == 'flow':
        # Use delta_psi if available, otherwise fallback to psi (though flow usually implies change)
        flow_input = delta_psi if delta_psi is not None else psi
        result['flow_data'] = calculate_flow_data(flow_input)
    
    if viz_type == 'phase_hsv':
        result['phase_hsv_data'] = calculate_phase_hsv_data(phase, density)
        
    if viz_type == 'complex_3d':
        result['complex_3d_data'] = calculate_complex_3d_data(real_part, imag_part)

    return result


def validate_psi(psi):
    if psi is None:
        raise ValueError("psi no puede ser None")
    if not isinstance(psi, torch.Tensor):
        raise TypeError(f"psi debe ser Tensor, recibido: {type(psi)}")

def normalize_psi_dimensions(psi: torch.Tensor) -> torch.Tensor:
    if psi.dim() == 4 and psi.shape[0] == 1:
        psi = psi.squeeze(0)
    return psi

def calculate_basic_quantities(psi):
    """
    Calcula density, phase, real, imag, energy.
    Soporta Tensor complejo [H, W, d] O QuantumStatePolar [B, 1, H, W].
    """
    is_polar = hasattr(psi, 'magnitude') and hasattr(psi, 'phase')
    
    with get_inference_context():
        if is_polar:
            # Polar Logic: Direct access!
            # Shape is (B, 1, H, W). We want (H, W).
            # Assume B=1
            mag = psi.magnitude.squeeze() # (H, W)
            pha = psi.phase.squeeze()     # (H, W)

            density = mag**2
            phase = pha
            real_part = mag * torch.cos(pha)
            imag_part = mag * torch.sin(pha)
            energy = density
        else:
            # Cartesian Logic
            psi_abs_sq = psi.abs()**2
            density = torch.sum(psi_abs_sq, dim=-1)

            if psi.shape[-1] > 0:
                phase_weighted = torch.angle(psi)
                phase_cos = torch.cos(phase_weighted).mean(dim=-1)
                phase_sin = torch.sin(phase_weighted).mean(dim=-1)
                phase = torch.atan2(phase_sin, phase_cos)
            else:
                phase = torch.angle(psi)[..., 0]

            real_part = psi.real
            imag_part = psi.imag
            energy = density

    return density, phase, real_part, imag_part, energy

def calculate_gradient_magnitude(density: np.ndarray):
    if len(density.shape) != 2:
        return np.zeros_like(density)
    grad_y, grad_x = np.gradient(density)
    return np.sqrt(grad_x**2 + grad_y**2)

def select_map_data(viz_type, density, phase, real_part, imag_part, gradient_magnitude, psi, motor=None):
    if viz_type == 'density':
        return density
    elif viz_type == 'phase':
        map_data = (phase + np.pi) / (2 * np.pi)
        return map_data
    elif viz_type == 'phase_hsv':
        # Handled in result construction for detailed data, but map_data needs to be something.
        # Usually frontend expects H or RGB. Return H normalized.
        return (phase + np.pi) / (2 * np.pi)
    elif viz_type == 'real':
        # Handle dimensionality
        if real_part.ndim == 3: return np.mean(real_part, axis=-1)
        return real_part
    elif viz_type == 'imag':
        # Parte imaginaria del primer canal o promedio
        if len(imag_part.shape) == 3:  # (H, W, d_state)
            return np.mean(imag_part, axis=-1) if imag_part.shape[-1] > 1 else imag_part[:, :, 0]
        elif len(imag_part.shape) == 2:
            return imag_part
        else:
            return imag_part.flatten().reshape(psi.shape[0], psi.shape[1]) if len(psi.shape) >= 2 else imag_part
    elif viz_type == 'gradient':
        return gradient_magnitude
    elif viz_type == 'spectral':
        # Transformada de Fourier (magnitud del espectro)
        if len(density.shape) == 2:
            fft = np.fft.fft2(density)
            fft_shifted = np.fft.fftshift(fft)
            spectral_magnitude = np.abs(fft_shifted)
            
            # Mask DC component (center pixel) to avoid it dominating the view
            # Replace with max of neighbors to preserve continuity
            cy, cx = spectral_magnitude.shape[0] // 2, spectral_magnitude.shape[1] // 2
            # Get 3x3 neighborhood around center
            neighborhood = spectral_magnitude[cy-1:cy+2, cx-1:cx+2]
            # Mask center
            mask = np.ones_like(neighborhood, dtype=bool)
            mask[1, 1] = False
            # Replace center with max of neighbors
            spectral_magnitude[cy, cx] = np.max(neighborhood[mask])
            
            # Log scale para mejor visualización
            return np.log1p(spectral_magnitude)
        else:
            return density
    elif viz_type == 'physics':
        # Mapa de física: muestra la "fuerza" de la interacción local (matriz A)
        if motor is not None and hasattr(motor, 'get_physics_matrix_map'):
            physics_map = motor.get_physics_matrix_map()
            if physics_map is not None:
                return physics_map
        # Fallback: usar densidad si no se puede calcular física
        return density
    elif viz_type == 'entropy':
        # Mapa de entropía: mide la complejidad/información por célula
        return calculate_entropy_map(psi)
    elif viz_type == 'coherence':
        # Mapa de coherencia: mide la coherencia de fase entre células vecinas
        return calculate_coherence_map(psi)
    elif viz_type == 'channel_activity':
        # Actividad por canal: muestra qué canales están más activos
        return calculate_channel_activity_map(psi)
    elif viz_type == 'fields':
        # Field Theory Visualization: Return first 3 channels as RGB
        # Channel 0: EM (Red), Channel 1: Gravity (Green), Channel 2: Higgs (Blue)
        if psi.shape[-1] >= 3:
            # Extract first 3 channels magnitude
            fields = psi.abs()[..., :3]  # (H, W, 3)
            return tensor_to_numpy(fields, "fields_map")
        elif psi.shape[-1] >= 1:
            # Fallback if less than 3 channels: Pad with zeros
            h, w = psi.shape[0], psi.shape[1]
            fields = torch.zeros((h, w, 3), device=psi.device, dtype=psi.dtype)
            # Copy available channels
            available = min(psi.shape[-1], 3)
            fields[..., :available] = psi.abs()[..., :available]
            return tensor_to_numpy(fields, "fields_map")
        else:
            return density
    else:
        return density

# Stub advanced calcs (Entropy, Coherence) for Polar for now to save space/time
def calculate_entropy_map(psi): return None
def calculate_coherence_map(psi): return None
def calculate_channel_activity_map(psi): return None

def calculate_histograms(density, phase, real, imag):
    # Simplified logic
    d_flat = density.flatten()
    return {'density': [{"bin": "0", "count": 0}]} # Stub
