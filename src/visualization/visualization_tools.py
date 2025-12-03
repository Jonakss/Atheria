# src/visualization_tools.py
import numpy as np
from io import BytesIO
import base64
import torch

def get_complex_parts(psi_tensor):
    """Divide un tensor complejo (real, imag) en dos tensores."""
    if psi_tensor is None or psi_tensor.shape[-1] % 2 != 0:
        return None, None
    d_state = psi_tensor.shape[-1] // 2
    real_parts = psi_tensor[..., :d_state]
    imag_parts = psi_tensor[..., d_state:]
    return real_parts, imag_parts

def get_density_map(psi, absolute_scale=True) -> np.ndarray:
    """Calcula el mapa de densidad (norma al cuadrado) de un estado psi."""
    if psi is None: return np.zeros((256, 256))

    # Manejar QuantumStatePolar u objetos similares si se pasa directamente
    if hasattr(psi, 'magnitude'):
         density = psi.magnitude.squeeze()**2
    else:
        real, imag = get_complex_parts(psi.cpu())
        if real is None: return np.zeros((psi.shape[1], psi.shape[2]))
        density = torch.sum(real**2 + imag**2, dim=-1)

    density = density.squeeze()
    if density.ndim == 3: # Handle batch if present
         density = density[0]

    if absolute_scale:
        # ESCALA FIJA: Asumimos que la energía máxima "interesante" es 1.0 (o lo que sea físico)
        # Esto hace que el ruido (0.001) se vea negro, y la materia (0.9) se vea brillante.
        norm_density = torch.clamp(density, 0, 1.0)
    else:
        # ESCALA RELATIVA (La vieja, que causa el ruido)
        mi, ma = density.min(), density.max()
        norm_density = (density - mi) / (ma - mi + 1e-8)

    return (norm_density * 255).cpu().numpy().astype(np.uint8)

def get_change_map(psi, prev_psi) -> np.ndarray:
    """Calcula la magnitud del cambio entre dos estados psi."""
    if psi is None or prev_psi is None: return np.zeros((256, 256))
    change = psi.cpu() - prev_psi.cpu()
    real, imag = get_complex_parts(change)
    if real is None: return np.zeros((psi.shape[1], psi.shape[2]))
    magnitude = torch.sum(real**2 + imag**2, dim=-1)
    return magnitude.squeeze(0).numpy()

def get_phase_map(psi):
    """
    Visualiza la fase del estado cuántico.
    """
    # Calcular el ángulo (fase) y convertirlo a un array de floats
    phase = torch.angle(psi[0]).cpu().numpy()
    
    # Normalizar de [-pi, pi] a [0, 255]
    phase_normalized = (phase + np.pi) / (2 * np.pi)
    return (phase_normalized * 255).astype(np.uint8)

def get_channels_map(psi):
    """
    Visualiza los primeros 3 canales del estado cuántico como un mapa RGB.
    Utiliza normalización compartida para preservar las proporciones de color.
    """
    if psi.shape[-1] < 3:
        # Si no hay suficientes canales, devolver un mapa negro
        return np.zeros((psi.shape[1], psi.shape[2], 3), dtype=np.uint8)

    # Tomar la magnitud de los primeros 3 canales
    channels = psi.abs()[0, :, :, :3].cpu().numpy()

    # --- ¡¡CORRECCIÓN CLAVE!! Normalización Compartida ---
    # 1. Encontrar el valor máximo global en los 3 canales
    max_val = channels.max()
    if max_val == 0:
        max_val = 1 # Evitar división por cero

    # 2. Normalizar todos los canales usando ese máximo global
    normalized_channels = channels / max_val
    
    # Convertir a uint8 para visualización
    img = (normalized_channels * 255).astype(np.uint8)
    return img

def fig_to_base64(fig):
    """Convierte una figura de Matplotlib a una imagen en base64."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
