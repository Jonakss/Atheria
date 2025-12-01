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


def get_visualization_data(psi: torch.Tensor, viz_type: str, delta_psi: torch.Tensor = None, motor=None, downsample_factor: int = 1):
    """
    Genera datos de visualización a partir del estado cuántico psi.
    
    Args:
        psi: Tensor complejo con el estado cuántico
        viz_type: Tipo de visualización ('density', 'phase', 'poincare', 'flow', 'phase_attractor', 'physics')
        delta_psi: Tensor opcional con delta_psi para visualizaciones de flujo
        motor: Instancia opcional de Aetheria_Motor para cálculos adicionales
        downsample_factor: Factor de downsampling para optimizar transferencia de datos
    
    Returns:
        Dict con map_data, hist_data, poincare_coords y datos adicionales según viz_type
    """
    # Validar entrada
    validate_psi(psi)
    
    # Normalizar dimensiones
    psi = normalize_psi_dimensions(psi)
    
    # DEBUG: Verificar que psi tenga datos válidos
    if isinstance(psi, torch.Tensor):
        psi_min = psi.abs().min().item() if psi.numel() > 0 else 0.0
        psi_max = psi.abs().max().item() if psi.numel() > 0 else 0.0
        psi_mean = psi.abs().mean().item() if psi.numel() > 0 else 0.0
        psi_shape = list(psi.shape)
        logging.debug(f"🔍 psi stats: shape={psi_shape}, min={psi_min:.6f}, max={psi_max:.6f}, mean={psi_mean:.6f}")
        
        # Verificar si psi está completamente vacío o en cero
        if psi_max < 1e-10:
            logging.warning(f"⚠️ psi tiene valores muy pequeños (max={psi_max:.6e}). Puede estar vacío o no inicializado correctamente.")
    
    # Aplicar downsampling si se especifica
    psi = apply_downsampling(psi, downsample_factor)
    
    # Sincronizar GPU si es necesario
    device = psi.device if isinstance(psi, torch.Tensor) else torch.device('cpu')
    synchronize_gpu(device)
    
    # Calcular cantidades básicas en GPU
    density, phase, real_part, imag_part, energy = calculate_basic_quantities(psi)
    
    # DEBUG: Verificar que density tenga datos válidos
    if isinstance(density, torch.Tensor):
        density_min = density.min().item() if density.numel() > 0 else 0.0
        density_max = density.max().item() if density.numel() > 0 else 0.0
        density_mean = density.mean().item() if density.numel() > 0 else 0.0
        logging.debug(f"🔍 density stats: shape={list(density.shape)}, min={density_min:.6f}, max={density_max:.6f}, mean={density_mean:.6f}")
        
        # Verificar si density está vacío
        if density_max < 1e-10:
            logging.error(f"❌ density tiene valores muy pequeños (max={density_max:.6e}). Esto causará visualización en gris.")
    
    # Mover todo a CPU en batch (una sola sincronización CUDA)
    density = tensor_to_numpy(density, "density")
    phase = tensor_to_numpy(phase, "phase")
    real_part = tensor_to_numpy(real_part, "real_part")
    imag_part = tensor_to_numpy(imag_part, "imag_part")
    energy = tensor_to_numpy(energy, "energy")
    
    # Calcular gradiente espacial
    gradient_magnitude = calculate_gradient_magnitude(density)
    
    # Seleccionar map_data según viz_type
    map_data = select_map_data(
        viz_type, density, phase, real_part, imag_part, 
        gradient_magnitude, psi, motor
    )
    
    # Verificar que map_data no esté vacío
    if map_data is None or (hasattr(map_data, 'size') and map_data.size == 0):
        logging.warning(f"⚠️ map_data está vacío para viz_type={viz_type}. Usando densidad como fallback.")
        map_data = density
    
    # Verificar que map_data tenga la forma correcta (debe ser 2D)
    if hasattr(map_data, 'shape') and len(map_data.shape) != 2:
        logging.warning(f"⚠️ map_data tiene forma incorrecta {map_data.shape} para viz_type={viz_type}. Reshapeando.")
        try:
            if map_data.size > 0:
                # Intentar reshapear a 2D basado en las dimensiones de psi
                h, w = psi.shape[0], psi.shape[1]
                map_data = map_data.flatten()[:h*w].reshape(h, w)
            else:
                map_data = density
        except Exception as e:
            logging.error(f"Error reshapeando map_data: {e}. Usando densidad como fallback.")
            map_data = density
    
    # Normalizar map_data a [0, 1]
    # Nota: Algunos tipos de visualización (como 'phase') ya normalizan los datos,
    # pero normalizamos de nuevo para asegurar consistencia
    map_data = normalize_map_data(map_data)
    
    # Logging para debugging (siempre activo para diagnóstico)
    if map_data.size > 0:
        min_val, max_val = np.min(map_data), np.max(map_data)
        mean_val = np.mean(map_data)
        std_val = np.std(map_data)
        range_val = max_val - min_val
        
        logging.info(f"📊 map_data stats para viz_type={viz_type}: shape={map_data.shape}, min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}, range={range_val:.6f}")
        
        # Verificar si está en rango [0, 1] esperado (normalizado)
        if min_val >= 0 and max_val <= 1:
            logging.debug(f"✅ map_data está normalizado correctamente a [0, 1] para viz_type={viz_type}")
        else:
            logging.warning(f"⚠️ map_data NO está en rango [0, 1] esperado (min={min_val:.6f}, max={max_val:.6f}) para viz_type={viz_type}")
        
        if range_val < 1e-6:
            logging.warning(f"⚠️ map_data normalizado tiene rango muy pequeño ({min_val:.6f} - {max_val:.6f}) para viz_type={viz_type}. Todos los valores son muy similares. Esto causará visualización uniforme.")
        
        # Verificar si todos los valores son exactamente 0.5 (gris medio de fallback)
        if abs(mean_val - 0.5) < 1e-3 and abs(std_val) < 1e-3:
            logging.warning(f"⚠️ map_data parece estar en estado de fallback (todos ~0.5). Los datos originales pueden estar vacíos o todos iguales. Verificar que el motor tenga estado válido.")
    else:
        logging.error(f"❌ map_data está vacío después de normalización para viz_type={viz_type}!")
    
    # Inicializar resultado
    # CRÍTICO: NO convertir a .tolist() aquí - es muy lento para arrays grandes (256x256)
    # optimize_frame_payload() se encargará de la conversión eficiente
    result = {
        "map_data": map_data,  # Dejar como numpy array
        "hist_data": {},
        "poincare_coords": [[0.0, 0.0]]
    }
    
    # --- Cálculo de datos para Poincaré (solo si se necesita) ---
    if viz_type in ['poincare', 'poincare_3d']:
        poincare_coords = calculate_poincare_coords(psi)
        result["poincare_coords"] = poincare_coords
    
    # --- Histogramas (solo si se necesitan) ---
    if viz_type == 'histogram':
        result["hist_data"] = calculate_histograms(density, phase, real_part, imag_part)
    
    # --- Datos para visualización 3D compleja (real vs imag vs tiempo) ---
    if viz_type == 'complex_3d':
        result["complex_3d_data"] = calculate_complex_3d_data(real_part, imag_part)
    
    # --- Datos HSV si se calculó phase_hsv ---
    if viz_type == 'phase_hsv':
        result['phase_hsv_data'] = calculate_phase_hsv_data(phase, density)
    
    # --- Phase Attractor: Solo calcular si se necesita ---
    if viz_type == 'phase_attractor' and psi.shape[-1] >= 2:
        result["phase_attractor"] = calculate_phase_attractor(psi)
    else:
        result["phase_attractor"] = None
    
    # --- Flow Viewer: Solo calcular si se necesita (viz_type == 'flow') ---
    if delta_psi is not None and viz_type == 'flow':
        result["flow_data"] = calculate_flow_data(delta_psi)
    else:
        result["flow_data"] = None
    
    # --- Phase Space (PCA + KMeans) ---
    if viz_type == 'phase_space':
        result["phase_space_data"] = get_phase_space_data(psi)
    
    return result


def validate_psi(psi: torch.Tensor):
    """
    Valida que psi tenga el formato correcto.
    
    Raises:
        ValueError: Si psi no es válido
        TypeError: Si psi no es un tensor
    """
    if psi is None:
        raise ValueError("psi no puede ser None")
    
    if not isinstance(psi, torch.Tensor):
        raise TypeError(f"psi debe ser un torch.Tensor, recibido: {type(psi)}")
    
    # Validar dimensiones
    if psi.dim() < 3:
        raise ValueError(f"psi debe tener al menos 3 dimensiones, recibido: {psi.dim()}")
    
    # Validar que psi tenga la última dimensión (d_state)
    if psi.shape[-1] == 0:
        raise ValueError("psi no puede tener d_state = 0")


def normalize_psi_dimensions(psi: torch.Tensor) -> torch.Tensor:
    """
    Normaliza las dimensiones de psi (elimina batch dimension si es 1).
    
    Args:
        psi: Tensor complejo con el estado cuántico
    
    Returns:
        Tensor normalizado [H, W, d_state]
    """
    # Normalizar dimensiones
    if psi.dim() == 4 and psi.shape[0] == 1:
        psi = psi.squeeze(0)
    
    return psi


def calculate_basic_quantities(psi: torch.Tensor):
    """
    Calcula las cantidades básicas (density, phase, real, imag, energy) en GPU.
    
    Args:
        psi: Tensor complejo con el estado cuántico [H, W, d_state]
    
    Returns:
        Tuple de (density, phase, real_part, imag_part, energy) como tensores
    """
    device = psi.device if isinstance(psi, torch.Tensor) else torch.device('cpu')
    
    with get_inference_context():
        # Calcular todo en GPU vectorizado
        psi_abs_sq = psi.abs()**2  # |ψ|²
        density = torch.sum(psi_abs_sq, dim=-1)
        
        # Calcular fase en GPU (vectorizado)
        if psi.shape[-1] > 0:
            # Calcular promedio circular de fases en GPU (más eficiente)
            phase_weighted = torch.angle(psi)
            phase_cos = torch.cos(phase_weighted).mean(dim=-1)
            phase_sin = torch.sin(phase_weighted).mean(dim=-1)
            phase = torch.atan2(phase_sin, phase_cos)
        else:
            phase = torch.angle(psi)
            if phase.ndim > 2:
                phase = phase[..., 0]
        
        # Calcular partes real e imaginaria (ya en GPU)
        real_part = psi.real
        imag_part = psi.imag
        
        # Calcular energía total (suma de |ψ|² sobre todos los canales)
        energy = density  # Ya calculado arriba
    
    return density, phase, real_part, imag_part, energy


def calculate_gradient_magnitude(density: np.ndarray):
    """
    Calcula la magnitud del gradiente espacial de la densidad.
    
    Args:
        density: Array 2D de numpy con densidad
    
    Returns:
        Array 2D con magnitud del gradiente
    """
    if len(density.shape) != 2:
        return np.zeros_like(density)
    
    # Intentar calcular en GPU primero
    try:
        density_tensor = torch.tensor(density) if not isinstance(density, torch.Tensor) else density
        if isinstance(density_tensor, torch.Tensor):
            # Calcular gradiente en GPU usando torch.diff
            grad_y_torch = torch.diff(density_tensor, dim=0, prepend=density_tensor[0:1, :])
            grad_x_torch = torch.diff(density_tensor, dim=1, prepend=density_tensor[:, 0:1])
            gradient_magnitude_tensor = torch.sqrt(grad_x_torch**2 + grad_y_torch**2)
            return gradient_magnitude_tensor.cpu().numpy()
    except Exception:
        pass
    
    # Fallback a numpy si falla GPU
    grad_y, grad_x = np.gradient(density)
    return np.sqrt(grad_x**2 + grad_y**2)


def select_map_data(viz_type: str, density: np.ndarray, phase: np.ndarray, 
                   real_part: np.ndarray, imag_part: np.ndarray, 
                   gradient_magnitude: np.ndarray, psi: torch.Tensor, motor=None) -> np.ndarray:
    """
    Selecciona los datos del mapa según el tipo de visualización.
    
    Args:
        viz_type: Tipo de visualización
        density: Array con densidad
        phase: Array con fase
        real_part: Array con parte real
        imag_part: Array con parte imaginaria
        gradient_magnitude: Array con magnitud del gradiente
        psi: Tensor complejo original
        motor: Instancia opcional de motor para cálculos adicionales
    
    Returns:
        Array con map_data según el tipo de visualización
    """
    if viz_type == 'density':
        return density
    elif viz_type == 'phase':
        # Normalizar fase a [0, 1] con mejor contraste
        map_data = (phase + np.pi) / (2 * np.pi)
        # Aplicar un estiramiento de contraste para mejor visibilidad
        p2, p98 = np.percentile(map_data, [2, 98])
        if p98 > p2:
            map_data = np.clip((map_data - p2) / (p98 - p2), 0, 1)
        return map_data
    elif viz_type == 'phase_hsv':
        # Fase en HSV: H = fase, S = 1, V = densidad normalizada
        phase_normalized = (phase + np.pi) / (2 * np.pi)  # [0, 1]
        density_norm = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-10)
        # Combinar fase (hue) con densidad (value) - se convertirá a RGB en el frontend
        return phase_normalized  # El frontend usará esto como hue
    elif viz_type == 'energy':
        return density  # Energy es igual a density
    elif viz_type == 'real':
        # Parte real del primer canal o promedio
        if len(real_part.shape) == 3:  # (H, W, d_state)
            return np.mean(real_part, axis=-1) if real_part.shape[-1] > 1 else real_part[:, :, 0]
        elif len(real_part.shape) == 2:
            return real_part
        else:
            return real_part.flatten().reshape(psi.shape[0], psi.shape[1]) if len(psi.shape) >= 2 else real_part
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
    else:
        return density


def calculate_entropy_map(psi: torch.Tensor) -> np.ndarray:
    """Calcula mapa de entropía de Shannon en GPU."""
    try:
        # Calcular en GPU
        psi_abs_sq = psi.abs()**2  # (H, W, d_state)
        # Normalizar para obtener probabilidades
        total_prob = torch.sum(psi_abs_sq, dim=-1, keepdim=True)  # (H, W, 1)
        
        # Evitar división por cero
        probabilities = torch.where(total_prob > 1e-10, 
                                psi_abs_sq / (total_prob + 1e-10),
                                torch.tensor(1.0 / psi.shape[-1], device=psi.device))
        
        # Calcular entropía: -sum(p * log(p))
        # Añadir epsilon a log para estabilidad numérica
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)  # (H, W)
        
        # Normalizar por entropía máxima (log(d_state))
        max_entropy = np.log(psi.shape[-1])
        entropy_normalized = entropy / (max_entropy + 1e-10)
        
        return tensor_to_numpy(entropy_normalized, "entropy_map")
    except Exception as e:
        logging.warning(f"Error calculando entropía: {e}")
        # Fallback a densidad
        density = torch.sum(psi.abs()**2, dim=-1)
        return tensor_to_numpy(density)


def calculate_coherence_map(psi: torch.Tensor) -> np.ndarray:
    """Calcula mapa de coherencia de fase en GPU."""
    try:
        # Calcular en GPU
        # Calcular coherencia horizontal (entre vecinos en x)
        psi_shifted_x = torch.roll(psi, shifts=-1, dims=1)  # Desplazar en x
        
        # Ajustar borde (última columna no tiene vecino derecho)
        # Clonar para evitar modificar tensor original in-place si es necesario
        # Pero roll crea copia, así que podemos modificar psi_shifted_x
        psi_shifted_x[:, -1, :] = 0
        
        # Producto interno: ⟨ψ | ψ_shifted⟩ = sum(ψ * conj(ψ_shifted))
        inner_product = torch.sum(psi * torch.conj(psi_shifted_x), dim=-1)  # (H, W)
        
        # Normas
        norm_psi = torch.sqrt(torch.sum(psi.abs()**2, dim=-1))  # (H, W)
        norm_shifted = torch.sqrt(torch.sum(psi_shifted_x.abs()**2, dim=-1))  # (H, W)
        
        # Coherencia: |⟨ψ | ψ_shifted⟩| / (|ψ| * |ψ_shifted|)
        coherence = inner_product.abs() / (norm_psi * norm_shifted + 1e-10)  # (H, W)
        
        return tensor_to_numpy(coherence, "coherence_map")
    except Exception as e:
        logging.warning(f"Error calculando coherencia: {e}")
        # Fallback a densidad
        density = torch.sum(psi.abs()**2, dim=-1)
        return tensor_to_numpy(density)


def calculate_channel_activity_map(psi: torch.Tensor) -> np.ndarray:
    """Calcula mapa de actividad por canal en GPU."""
    try:
        # Calcular en GPU
        psi_abs_sq = psi.abs()**2  # (H, W, d_state)
        
        # Calcular actividad promedio por canal (promedio espacial)
        # dim=(0, 1) reduce H y W
        channel_activity = torch.mean(psi_abs_sq, dim=(0, 1))  # (d_state,)
        
        # Encontrar el canal más activo para cada célula
        dominant_channel = torch.argmax(psi_abs_sq, dim=-1)  # (H, W) indices
        
        # Obtener el valor del canal dominante en cada posición
        # gather requiere dimensiones coincidentes, expandimos dominant_channel
        dominant_channel_expanded = dominant_channel.unsqueeze(-1)  # (H, W, 1)
        dominant_values = torch.gather(psi_abs_sq, -1, dominant_channel_expanded).squeeze(-1)  # (H, W)
        
        # Normalizar por actividad máxima global de canales
        max_activity = torch.max(channel_activity)
        
        if max_activity > 0:
            map_data = dominant_values / max_activity
            return tensor_to_numpy(map_data, "channel_activity_map")
        else:
            density = torch.sum(psi_abs_sq, dim=-1)
            return tensor_to_numpy(density)
    except Exception as e:
        logging.warning(f"Error calculando actividad de canales: {e}")
        # Fallback a densidad
        density = torch.sum(psi.abs()**2, dim=-1)
        return tensor_to_numpy(density)


def calculate_histograms(density: np.ndarray, phase: np.ndarray, 
                        real_part: np.ndarray, imag_part: np.ndarray) -> dict:
    """
    Calcula histogramas de las cantidades básicas.
    
    Args:
        density: Array con densidad
        phase: Array con fase
        real_part: Array con parte real
        imag_part: Array con parte imaginaria
    
    Returns:
        Dict con histogramas
    """
    density_flat = density.flatten()
    phase_flat = phase.flatten()
    real_flat = real_part.flatten()
    imag_flat = imag_part.flatten()

    density_hist, density_bins = np.histogram(density_flat, bins=30, range=(0, np.max(density_flat) if np.max(density_flat) > 0 else 1))
    phase_hist, phase_bins = np.histogram(phase_flat, bins=30, range=(-np.pi, np.pi))
    real_hist, real_bins = np.histogram(real_flat, bins=30, range=(-1, 1))
    imag_hist, imag_bins = np.histogram(imag_flat, bins=30, range=(-1, 1))

    return {
        'density': [{"bin": f"{density_bins[i]:.2f}", "count": int(density_hist[i])} for i in range(len(density_hist))],
        'phase': [{"bin": f"{phase_bins[i]:.2f}", "count": int(phase_hist[i])} for i in range(len(phase_hist))],
        'real': [{"bin": f"{real_bins[i]:.2f}", "count": int(real_hist[i])} for i in range(len(real_hist))],
        'imag': [{"bin": f"{imag_bins[i]:.2f}", "count": int(imag_hist[i])} for i in range(len(imag_hist))],
    }
