"""Utilidades para procesamiento y conversión de tensores en visualizaciones."""
import torch
import numpy as np
import logging


def apply_downsampling(psi: torch.Tensor, downsample_factor: int) -> torch.Tensor:
    """
    Aplica downsampling al tensor psi usando promedio (pooling promedio).
    
    Args:
        psi: Tensor complejo con el estado cuántico
        downsample_factor: Factor de reducción de tamaño
    
    Returns:
        Tensor downsampled
    """
    if downsample_factor <= 1:
        return psi
    
    H, W = psi.shape[0], psi.shape[1]
    new_H, new_W = H // downsample_factor, W // downsample_factor
    
    if new_H > 0 and new_W > 0:
        # Reshape y promedio
        psi_downsampled = psi[:new_H * downsample_factor, :new_W * downsample_factor].reshape(
            new_H, downsample_factor, new_W, downsample_factor, -1
        ).mean(dim=(1, 3))
        return psi_downsampled
    
    return psi


def tensor_to_numpy(tensor_or_array, name: str = "tensor"):
    """
    Convierte un tensor de PyTorch o array de numpy a numpy array.
    Maneja errores de conversión de manera robusta.
    
    Args:
        tensor_or_array: Tensor de PyTorch o array de numpy
        name: Nombre del tensor para logging
    
    Returns:
        Array de numpy
    """
    try:
        if isinstance(tensor_or_array, torch.Tensor) and hasattr(tensor_or_array, 'detach'):
            return tensor_or_array.detach().contiguous().cpu().numpy()
        elif not isinstance(tensor_or_array, np.ndarray):
            return np.array(tensor_or_array)
        else:
            return tensor_or_array
    except (AttributeError, TypeError) as e:
        logging.warning(f"Error convirtiendo {name}: {e}, intentando np.array()")
        try:
            return np.array(tensor_or_array)
        except Exception as e2:
            logging.error(f"Error crítico convirtiendo {name}: {e2}")
            raise


def synchronize_gpu(device: torch.device):
    """
    Sincroniza operaciones GPU si es necesario.
    
    Args:
        device: Dispositivo de PyTorch
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()


def get_inference_context():
    """
    Obtiene el contexto de inferencia adecuado (inference_mode o no_grad).
    
    Returns:
        Context manager para inferencia
    """
    if hasattr(torch, 'inference_mode'):
        return torch.inference_mode()
    else:
        return torch.no_grad()


def normalize_map_data(map_data: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
    """
    Normaliza map_data a [0, 1].
    
    IMPORTANTE: Si todos los valores son iguales (max == min), retorna 0.5 (gris medio)
    en lugar de 0 (negro) para indicar que hay datos pero sin variación.
    
    Args:
        map_data: Array de numpy con datos del mapa
        min_val: Valor mínimo (si None, usa np.min)
        max_val: Valor máximo (si None, usa np.max)
    
    Returns:
        Array normalizado a [0, 1]
    """
    if map_data.size == 0:
        return map_data
    
    if min_val is None:
        min_val = np.min(map_data)
    if max_val is None:
        max_val = np.max(map_data)
    
    # Si todos los valores son iguales, retornar 0.5 (gris medio) en lugar de ceros
    # Esto permite que el usuario vea que hay datos, aunque no haya variación
    if max_val <= min_val or abs(max_val - min_val) < 1e-10:
        # Si el valor común es 0 o muy cercano, usar 0.5
        # Si el valor común es distinto de 0, normalizar ese valor a 0.5
        if abs(max_val) < 1e-10:
            return np.full_like(map_data, 0.5, dtype=np.float32)
        else:
            # Todos los valores son iguales a max_val, mantener el valor pero normalizado a 0.5
            return np.full_like(map_data, 0.5, dtype=np.float32)
    
    # Normalización normal
    normalized = (map_data - min_val) / (max_val - min_val)
    # Asegurar que esté en [0, 1] y usar float32 para mejor rendimiento
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)

