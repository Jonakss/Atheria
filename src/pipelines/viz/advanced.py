"""Visualizaciones avanzadas: Poincaré, Flow, Phase Attractor, Complex 3D."""
import torch
import numpy as np
import logging
from sklearn.decomposition import PCA
from typing import Dict, Optional, Tuple, List

# Caché para cálculos de Poincaré (evitar recalcular en cada frame)
_poincare_cache = {
    'last_psi_hash': None,
    'last_coords': None,
    'recalc_counter': 0
}
POINCARE_RECALC_INTERVAL = 5  # Recalcular cada N frames

pca = PCA(n_components=2)


def calculate_poincare_coords(psi: torch.Tensor) -> List[List[float]]:
    """
    Calcula las coordenadas de Poincaré usando PCA.
    Usa caché y submuestreo para optimizar rendimiento.
    
    Args:
        psi: Tensor complejo con el estado cuántico [H, W, d_state]
    
    Returns:
        Lista de coordenadas [x, y] para cada punto
    """
    global _poincare_cache
    
    try:
        # OPTIMIZACIÓN: Usar caché y submuestreo para mejorar rendimiento
        # Crear hash simple del estado para detectar cambios
        psi_sample = psi[::max(1, psi.shape[0]//32), ::max(1, psi.shape[1]//32), :]  # Submuestreo para hash
        psi_hash = hash(psi_sample.cpu().numpy().tobytes())
        
        # Verificar caché
        use_cache = (
            _poincare_cache['last_psi_hash'] == psi_hash and 
            _poincare_cache['last_coords'] is not None
        )
        
        # Actualizar contador para recalcular periódicamente (aunque el hash no cambie)
        _poincare_cache['recalc_counter'] += 1
        should_recalc = _poincare_cache['recalc_counter'] >= POINCARE_RECALC_INTERVAL
        
        if use_cache and not should_recalc:
            # Usar caché
            poincare_coords = _poincare_cache['last_coords']
            logging.debug(f"Poincaré: usando caché (hash={psi_hash})")
            return poincare_coords
        
        # Calcular Poincaré con optimizaciones
        # OPTIMIZACIÓN 1: Submuestreo inteligente (solo usar cada N-ésimo punto para PCA)
        subsample_factor = max(1, int(np.sqrt(psi.shape[0] * psi.shape[1] / 10000)))  # ~10k puntos máximo
        
        psi_subsampled = psi[::subsample_factor, ::subsample_factor, :]
        psi_flat_real = psi_subsampled.real.reshape(-1, psi_subsampled.shape[-1]).cpu().numpy()
        psi_flat_imag = psi_subsampled.imag.reshape(-1, psi_subsampled.shape[-1]).cpu().numpy()
        psi_flat_for_pca = np.concatenate([psi_flat_real, psi_flat_imag], axis=1)
        
        # OPTIMIZACIÓN 2: Limitar número máximo de puntos para PCA
        max_points = 5000
        if psi_flat_for_pca.shape[0] > max_points:
            # Seleccionar puntos aleatorios
            indices = np.random.choice(psi_flat_for_pca.shape[0], max_points, replace=False)
            psi_flat_for_pca = psi_flat_for_pca[indices]
        
        # Validar que hay suficientes puntos para PCA
        if psi_flat_for_pca.shape[0] >= 2:
            # OPTIMIZACIÓN 3: Usar fit_transform solo si cambió el estado significativamente
            poincare_coords = pca.fit_transform(psi_flat_for_pca)
            max_abs_val = np.max(np.abs(poincare_coords))
            if max_abs_val > 0:
                poincare_coords = poincare_coords / max_abs_val
            
            # Actualizar caché
            _poincare_cache['last_psi_hash'] = psi_hash
            _poincare_cache['last_coords'] = poincare_coords.copy()
            _poincare_cache['recalc_counter'] = 0
            logging.debug(f"Poincaré: recalculado (subsample={subsample_factor}, puntos={psi_flat_for_pca.shape[0]})")
            
            return poincare_coords.tolist()
        else:
            logging.warning(f"Poincaré: no hay suficientes puntos ({psi_flat_for_pca.shape[0]})")
            return [[0.0, 0.0]]
            
    except Exception as e:
        logging.warning(f"Error al calcular coordenadas de Poincaré: {e}. Usando coordenadas por defecto.", exc_info=True)
        return [[0.0, 0.0]]


def calculate_phase_attractor(psi: torch.Tensor) -> Optional[Dict]:
    """
    Calcula datos del atractor de fase (valores de los canales en el centro).
    
    Args:
        psi: Tensor complejo con el estado cuántico [H, W, d_state]
    
    Returns:
        Dict con datos del phase attractor o None si psi no tiene suficientes canales
    """
    if psi.shape[-1] < 2:
        return None
    
    try:
        # Determinar dimensiones correctas
        if len(psi.shape) == 3:  # (H, W, d_state)
            h, w = psi.shape[0], psi.shape[1]
            center_y = h // 2
            center_x = w // 2
            psi_center = psi[center_y, center_x, :]
        elif len(psi.shape) == 4:  # (B, H, W, d_state)
            h, w = psi.shape[1], psi.shape[2]
            center_y = h // 2
            center_x = w // 2
            psi_center = psi[0, center_y, center_x, :]
        else:
            # Fallback para otras formas
            psi_center = psi[0, 0, :] if psi.shape[-1] > 0 else psi[0]
        
        # Extraer canales 0 y 1
        channel_0 = psi_center[0].cpu().numpy()
        channel_1 = psi_center[1].cpu().numpy() if psi.shape[-1] > 1 else psi_center[0].cpu().numpy()
        
        return {
            "channel_0": {
                "real": float(channel_0.real),
                "imag": float(channel_0.imag),
                "abs": float(np.abs(channel_0))
            },
            "channel_1": {
                "real": float(channel_1.real),
                "imag": float(channel_1.imag),
                "abs": float(np.abs(channel_1))
            }
        }
    except Exception as e:
        logging.warning(f"Error calculando phase attractor: {e}")
        return None


def calculate_flow_data(delta_psi: torch.Tensor) -> Optional[Dict]:
    """
    Calcula datos de flujo a partir de delta_psi.
    
    Args:
        delta_psi: Tensor complejo con el cambio en el estado cuántico
    
    Returns:
        Dict con dx, dy, magnitude o None si hay error
    """
    try:
        # Convertir delta_psi a numpy
        if isinstance(delta_psi, torch.Tensor):
            delta_psi_np = delta_psi.cpu().numpy()
            # Normalizar dimensiones
            if delta_psi_np.ndim == 4 and delta_psi_np.shape[0] == 1:
                delta_psi_np = delta_psi_np.squeeze(0)
        else:
            delta_psi_np = delta_psi
        
        # Calcular dirección y magnitud del flujo
        if delta_psi_np.shape[-1] >= 2:
            # Usar canales 0 y 1 para dirección (promediar todos los canales si hay más)
            if len(delta_psi_np.shape) == 3:  # (H, W, d_state)
                # Promediar todos los canales para obtener un campo vectorial más robusto
                dx_all = delta_psi_np[:, :, :].real  # (H, W, d_state)
                dy_all = delta_psi_np[:, :, :].imag  # (H, W, d_state)
                # Calcular componentes x,y promediando magnitudes
                dx = np.mean(dx_all, axis=-1)  # (H, W)
                dy = np.mean(dy_all, axis=-1)  # (H, W)
                magnitude = np.mean(np.abs(delta_psi_np), axis=-1)  # (H, W)
            elif len(delta_psi_np.shape) == 4:  # (B, H, W, d_state)
                dx_all = delta_psi_np[0, :, :, :].real  # (H, W, d_state)
                dy_all = delta_psi_np[0, :, :, :].imag  # (H, W, d_state)
                dx = np.mean(dx_all, axis=-1)  # (H, W)
                dy = np.mean(dy_all, axis=-1)  # (H, W)
                magnitude = np.mean(np.abs(delta_psi_np[0, :, :, :]), axis=-1)  # (H, W)
            else:
                dx = np.zeros((10, 10))
                dy = np.zeros((10, 10))
                magnitude = np.zeros((10, 10))
            
            # Normalizar magnitudes para visualización (usar percentiles para mejor contraste)
            mag_flat = magnitude.flatten()
            p95 = np.percentile(mag_flat, 95) if len(mag_flat) > 0 else 1.0
            max_mag = max(np.max(magnitude), p95) if np.max(magnitude) > 0 else 1.0
            magnitude_norm = np.clip(magnitude / max_mag, 0, 1)
            
            # Normalizar dx y dy para que las flechas tengan longitud visible
            max_dx = np.max(np.abs(dx)) if np.max(np.abs(dx)) > 0 else 1.0
            max_dy = np.max(np.abs(dy)) if np.max(np.abs(dy)) > 0 else 1.0
            max_component = max(max_dx, max_dy)
            
            # Escalar dx y dy para mejor visualización (mantener proporción)
            scale_factor = 1.0 / max_component if max_component > 0 else 1.0
            dx_scaled = dx * scale_factor
            dy_scaled = dy * scale_factor
            
            return {
                "dx": dx_scaled.tolist(),
                "dy": dy_scaled.tolist(),
                "magnitude": magnitude_norm.tolist()
            }
        else:
            return None
    except Exception as e:
        logging.warning(f"Error calculando datos de flujo: {e}")
        return None


def calculate_complex_3d_data(real_part: np.ndarray, imag_part: np.ndarray) -> Dict:
    """
    Calcula datos 3D complejos (real vs imag).
    
    Args:
        real_part: Array con parte real [H, W, d_state] o [H, W]
        imag_part: Array con parte imaginaria [H, W, d_state] o [H, W]
    
    Returns:
        Dict con real e imag promediados sobre canales
    """
    try:
        # Extraer parte real e imaginaria promediadas sobre canales
        if len(real_part.shape) == 3:  # (H, W, d_state)
            real_avg = np.mean(real_part, axis=-1) if real_part.shape[-1] > 1 else real_part[:, :, 0]
        else:
            real_avg = real_part
        if len(imag_part.shape) == 3:  # (H, W, d_state)
            imag_avg = np.mean(imag_part, axis=-1) if imag_part.shape[-1] > 1 else imag_part[:, :, 0]
        else:
            imag_avg = imag_part
        
        return {
            "real": real_avg.tolist(),
            "imag": imag_avg.tolist()
        }
    except Exception as e:
        logging.warning(f"Error calculando complex_3d_data: {e}")
        # Fallback a arrays vacíos
        return {
            "real": [],
            "imag": []
        }


def calculate_phase_hsv_data(phase: np.ndarray, density: np.ndarray) -> Dict:
    """
    Calcula datos HSV combinando fase (hue) con densidad (value).
    
    Args:
        phase: Array con fase
        density: Array con densidad
    
    Returns:
        Dict con hue, saturation, value
    """
    try:
        phase_normalized = (phase + np.pi) / (2 * np.pi)  # [0, 1]
        density_norm = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-10)
        
        return {
            'hue': phase_normalized.tolist(),
            'saturation': np.ones_like(phase_normalized).tolist(),
            'value': density_norm.tolist()
        }
    except Exception as e:
        logging.warning(f"Error calculando phase_hsv_data: {e}")
        # Fallback
        return {
            'hue': [],
            'saturation': [],
            'value': []
        }

