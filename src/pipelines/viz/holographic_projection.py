"""
Proyección Holográfica Genérica

Este módulo implementa la técnica de proyección holográfica (AdS/CFT inspirada)
para convertir CUALQUIER estado 2D en un volumen 3D usando Scale-Space.

La idea: Un campo cuántico 2D contiene información suficiente para reconstruir
una representación 3D donde la "profundidad" representa la escala de renormalización.
"""

import torch
import torch.nn.functional as F
import logging

def project_2d_to_3d_holographic(
    state_2d: torch.Tensor,
    depth: int = 8,
    sigma_scale: float = 0.5,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Proyecta un estado 2D a un volumen 3D usando Scale-Space (AdS/CFT).
    
    Concepto:
    - Capa Z=0: Estado original (UV - alta frecuencia, detalles finos)
    - Capas Z>0: Versiones suavizadas (IR - baja frecuencia, estructuras gruesas)
    - La profundidad Z representa la escala de renormalización
    
    Args:
        state_2d: Tensor [B, H, W, C] o [H, W, C] o [B, H, W]
                 Magnitud o fase del campo cuántico 2D
        depth: Número de capas a generar (profundidad del bulk)
        sigma_scale: Factor de escala para sigma (0.5 = suave, 2.0 = agresivo)
        device: Dispositivo ('cpu' o 'cuda')
    
    Returns:
        Tensor [B, D, H, W] donde D=depth
        Volumen 3D listo para HolographicVolumeViewer
    
    Ejemplo:
        >>> psi = motor.state.psi  # [1, 64, 64, 8]
        >>> magnitude = torch.sqrt(torch.sum(psi.abs().pow(2), dim=-1, keepdim=True))  # [1, 64, 64, 1]
        >>> volume = project_2d_to_3d_holographic(magnitude, depth=8)  # [1, 8, 64, 64]
    """
    
    # Normalizar dimensiones
    if state_2d.dim() == 3:
        # [H, W, C] -> [1, H, W, C]
        state_2d = state_2d.unsqueeze(0)
    elif state_2d.dim() == 2:
        # [H, W] -> [1, H, W, 1]
        state_2d = state_2d.unsqueeze(0).unsqueeze(-1)
    
    # Si tiene múltiples canales (d_state), colapsar a magnitud
    if state_2d.shape[-1] > 1:
        # Asumir que es complejo: calcular magnitud
        magnitude = torch.sqrt(torch.sum(state_2d.abs().pow(2), dim=-1, keepdim=True))
    else:
        magnitude = state_2d
    
    # [B, H, W, 1] -> [B, 1, H, W] para conv2d
    magnitude = magnitude.permute(0, 3, 1, 2)
    
    # Asegurar que esté en el dispositivo correcto
    magnitude = magnitude.to(device)
    
    bulk_layers = []
    
    # Capa 0: Boundary (estado original)
    bulk_layers.append(magnitude)
    
    # Capas 1 a depth-1: Bulk (renormalización)
    for z in range(1, depth):
        # Sigma crece con la profundidad (más suavizado = mayor escala)
        sigma = z * sigma_scale + 0.5
        
        # Tamaño del kernel (debe ser impar)
        kernel_size = int(sigma * 4) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Aplicar Gaussian blur
        blurred = gaussian_blur_2d(magnitude, kernel_size, sigma)
        bulk_layers.append(blurred)
    
    # Concatenar capas: [B, 1, H, W] * depth -> [B, depth, H, W]
    bulk_volume = torch.cat(bulk_layers, dim=1)
    
    logging.debug(f"🔮 Proyección holográfica: {state_2d.shape} -> {bulk_volume.shape}")
    
    return bulk_volume


def gaussian_blur_2d(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Aplica Gaussian blur 2D a una imagen.
    
    Args:
        img: Tensor [B, C, H, W]
        kernel_size: Tamaño del kernel (debe ser impar)
        sigma: Desviación estándar del Gaussian
    
    Returns:
        Tensor [B, C, H, W] suavizado
    """
    # Crear kernel Gaussian 1D
    x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    
    # Kernel 2D = producto externo de 1D
    kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    
    # Aplicar convolución
    # Padding para mantener tamaño
    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel, padding=padding, groups=1)
    
    return blurred


# Función de conveniencia para uso directo
def visualize_as_hologram(motor, depth: int = 8, use_phase: bool = False):
    """
    Wrapper de conveniencia para visualizar cualquier motor como holograma 3D.
    
    Args:
        motor: Motor con atributo state.psi
        depth: Profundidad del volumen a generar
        use_phase: Si True, proyecta la fase en vez de la magnitud
    
    Returns:
        Tensor [1, D, H, W] listo para HolographicVolumeViewer
    """
    psi = motor.state.psi
    
    if use_phase:
        # Proyectar la fase
        phase = torch.angle(psi)  # [B, H, W, C]
        # Normalizar a [0, 1] para visualización
        phase_normalized = (phase + torch.pi) / (2 * torch.pi)
        source = phase_normalized
    else:
        # Proyectar la magnitud (default)
        source = psi.abs()
    
    device = psi.device
    volume = project_2d_to_3d_holographic(source, depth=depth, device=device)
    
    return volume
