# src/visualization_tools.py
import numpy as np
import torch

def get_complex_parts(psi_tensor: torch.Tensor):
    """Divide un tensor complejo (real, imag) en dos tensores."""
    if psi_tensor is None or psi_tensor.shape[-1] % 2 != 0:
        return None, None
    d_state = psi_tensor.shape[-1] // 2
    real_parts = psi_tensor[..., :d_state]
    imag_parts = psi_tensor[..., d_state:]
    return real_parts, imag_parts

def get_density_map(psi: torch.Tensor) -> np.ndarray:
    """Calcula el mapa de densidad (norma al cuadrado) de un estado psi."""
    if psi is None: return np.zeros((256, 256))
    real, imag = get_complex_parts(psi.cpu())
    if real is None: return np.zeros((psi.shape[1], psi.shape[2]))
    density = torch.sum(real**2 + imag**2, dim=-1)
    return density.squeeze(0).numpy()

def get_change_map(psi: torch.Tensor, prev_psi: torch.Tensor) -> np.ndarray:
    """Calcula la magnitud del cambio entre dos estados psi."""
    if psi is None or prev_psi is None: return np.zeros((256, 256))
    change = psi.cpu() - prev_psi.cpu()
    real, imag = get_complex_parts(change)
    if real is None: return np.zeros((psi.shape[1], psi.shape[2]))
    magnitude = torch.sum(real**2 + imag**2, dim=-1)
    return magnitude.squeeze(0).numpy()

def get_phase_map(psi: torch.Tensor) -> np.ndarray:
    """Calcula el mapa de fase (ángulo promedio) de un estado psi."""
    if psi is None: return np.zeros((256, 256))
    real, imag = get_complex_parts(psi.cpu())
    if real is None: return np.zeros((psi.shape[1], psi.shape[2]))
    # Calcular el ángulo para cada componente y luego promediar
    angles = torch.atan2(imag, real)
    mean_angle = torch.mean(angles, dim=-1)
    return mean_angle.squeeze(0).numpy()

def get_channels_map(psi: torch.Tensor) -> np.ndarray:
    """Mapea los primeros 3 componentes de densidad a canales RGB."""
    if psi is None: return np.zeros((256, 256, 3))
    real, imag = get_complex_parts(psi.cpu())
    if real is None: return np.zeros((psi.shape[1], psi.shape[2], 3))
    
    density_per_channel = real**2 + imag**2
    
    # Toma hasta 3 canales para RGB
    num_channels = density_per_channel.shape[-1]
    r = density_per_channel[..., 0] if num_channels > 0 else torch.zeros_like(density_per_channel[..., 0])
    g = density_per_channel[..., 1] if num_channels > 1 else torch.zeros_like(density_per_channel[..., 0])
    b = density_per_channel[..., 2] if num_channels > 2 else torch.zeros_like(density_per_channel[..., 0])
    
    # Apila y normaliza cada canal por separado para un mejor contraste
    rgb = np.stack([
        (r - r.min()) / (r.max() - r.min() + 1e-8),
        (g - g.min()) / (g.max() - g.min() + 1e-8),
        (b - b.min()) / (b.max() - b.min() + 1e-8)
    ], axis=-1)
    
    return (rgb * 255).squeeze(0).numpy().astype(np.uint8)

def fig_to_base64(fig):
    """Convierte una figura de Matplotlib a una imagen en base64."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
