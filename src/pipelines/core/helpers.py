"""Funciones helper para el pipeline server."""
import math
from typing import Tuple, Optional


def calculate_adaptive_downsample(grid_size: int, max_visualization_size: int = 512) -> int:
    """
    Calcula el factor de downsampling adaptativo basado en el tamaño del grid.
    
    Estrategia:
    - Si grid_size <= max_visualization_size: No downsampling (factor = 1)
    - Si grid_size > max_visualization_size: Downsample para mantener ~max_visualization_size píxeles
    - Factor mínimo: 1, Factor máximo: calculado para mantener rendimiento
    
    Args:
        grid_size: Tamaño del grid (H x W)
        max_visualization_size: Tamaño máximo deseado para visualización (default: 512)
    
    Returns:
        Factor de downsampling (1, 2, 4, 8, etc.)
    """
    if grid_size <= max_visualization_size:
        return 1
    
    # Calcular factor de downsampling para mantener ~max_visualization_size
    # Factor debe ser potencia de 2 para mejor rendimiento (2, 4, 8, 16...)
    factor = max(2, int(grid_size / max_visualization_size))
    
    # Redondear a la potencia de 2 más cercana hacia arriba
    factor = 2 ** math.ceil(math.log2(factor))
    
    # Límite máximo razonable (downsampling de 16x es bastante agresivo)
    factor = min(factor, 16)
    
    return factor


def calculate_adaptive_roi(grid_size: int, default_roi_size: int = 256) -> Optional[Tuple[int, int, int, int]]:
    """
    Calcula ROI adaptativo para grids grandes.
    
    Para grids grandes (>512), sugerir ROI automático centrado para reducir overhead.
    
    Args:
        grid_size: Tamaño del grid
        default_roi_size: Tamaño del ROI deseado (default: 256)
    
    Returns:
        ROI tuple (x, y, width, height) o None si no se necesita
    """
    # Solo aplicar ROI automático para grids muy grandes
    if grid_size <= 512:
        return None
    
    # Calcular ROI centrado
    roi_size = min(default_roi_size, grid_size)
    x = (grid_size - roi_size) // 2
    y = (grid_size - roi_size) // 2
    
    return (x, y, roi_size, roi_size)

