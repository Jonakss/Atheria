# src/visualization_tools.py
import numpy as np
from io import BytesIO
import base64

def get_complex_parts(psi_tensor):
    """Divide un tensor complejo (real, imag) en dos tensores."""
    import torch
    if psi_tensor is None or psi_tensor.shape[-1] % 2 != 0:
        return None, None
    d_state = psi_tensor.shape[-1] // 2
    real_parts = psi_tensor[..., :d_state]
    imag_parts = psi_tensor[..., d_state:]
    return real_parts, imag_parts

def get_density_map(psi) -> np.ndarray:
    """Calcula el mapa de densidad (norma al cuadrado) de un estado psi."""
    import torch
    if psi is None: return np.zeros((256, 256))
    real, imag = get_complex_parts(psi.cpu())
    if real is None: return np.zeros((psi.shape[1], psi.shape[2]))
    density = torch.sum(real**2 + imag**2, dim=-1)
    return density.squeeze(0).numpy()

def get_change_map(psi, prev_psi) -> np.ndarray:
    """Calcula la magnitud del cambio entre dos estados psi."""
    import torch
    if psi is None or prev_psi is None: return np.zeros((256, 256))
    change = psi.cpu() - prev_psi.cpu()
    real, imag = get_complex_parts(change)
    if real is None: return np.zeros((psi.shape[1], psi.shape[2]))
    magnitude = torch.sum(real**2 + imag**2, dim=-1)
    return magnitude.squeeze(0).numpy()

def get_phase_map(psi) -> np.ndarray:
    """Calcula el mapa de fase (ángulo promedio) de un estado psi."""
    import torch
    if psi is None: return np.zeros((256, 256))
    real, imag = get_complex_parts(psi.cpu())
    if real is None: return np.zeros((psi.shape[1], psi.shape[2]))
    # Calcular el ángulo para cada componente y luego promediar
    angles = torch.atan2(imag, real)
    mean_angle = torch.mean(angles, dim=-1)
    return mean_angle.squeeze(0).numpy()

def get_channels_map(psi) -> np.ndarray:
    """Mapea los primeros 3 componentes de densidad a canales RGB de forma segura."""
    import torch
    if psi is None: return np.zeros((256, 256, 3), dtype=np.uint8)
    
    real, imag = get_complex_parts(psi.cpu())
    if real is None: return np.zeros((psi.shape[1], psi.shape[2], 3), dtype=np.uint8)
    
    density_per_channel = real**2 + imag**2
    num_channels = density_per_channel.shape[-1]
    
    # Crea un canvas RGB vacío
    h, w = density_per_channel.shape[1], density_per_channel.shape[2]
    rgb_image = np.zeros((h, w, 3), dtype=np.float32)

    # Llena los canales que existen, normalizando cada uno individualmente
    if num_channels > 0:
        r = density_per_channel[0, ..., 0].numpy()
        rgb_image[..., 0] = (r - r.min()) / (r.max() - r.min() + 1e-8)
    if num_channels > 1:
        g = density_per_channel[0, ..., 1].numpy()
        rgb_image[..., 1] = (g - g.min()) / (g.max() - g.min() + 1e-8)
    if num_channels > 2:
        b = density_per_channel[0, ..., 2].numpy()
        rgb_image[..., 2] = (b - b.min()) / (b.max() - b.min() + 1e-8)
        
    return (rgb_image * 255).astype(np.uint8)

def fig_to_base64(fig):
    """Convierte una figura de Matplotlib a una imagen en base64."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
