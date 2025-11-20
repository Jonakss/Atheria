# src/pipeline_viz.py
import torch
import numpy as np
from sklearn.decomposition import PCA
import logging

pca = PCA(n_components=2)

# Caché para cálculos de Poincaré (evitar recalcular en cada frame)
_poincare_cache = {
    'last_psi_hash': None,
    'last_coords': None,
    'recalc_counter': 0
}
POINCARE_RECALC_INTERVAL = 5  # Recalcular cada N frames

def get_visualization_data(psi: torch.Tensor, viz_type: str, delta_psi: torch.Tensor = None, motor=None, downsample_factor: int = 1):
    """
    Genera datos de visualización a partir del estado cuántico psi.
    
    Args:
        psi: Tensor complejo con el estado cuántico
        viz_type: Tipo de visualización ('density', 'phase', 'poincare', 'flow', 'phase_attractor', 'physics')
        delta_psi: Tensor opcional con delta_psi para visualizaciones de flujo
        motor: Instancia opcional de Aetheria_Motor para cálculos adicionales
    
    Returns:
        Dict con map_data, hist_data, poincare_coords y datos adicionales según viz_type
    """
    # Validar entrada
    if psi is None:
        raise ValueError("psi no puede ser None")
    
    if not isinstance(psi, torch.Tensor):
        raise TypeError(f"psi debe ser un torch.Tensor, recibido: {type(psi)}")
    
    # Validar dimensiones
    if psi.dim() < 3:
        raise ValueError(f"psi debe tener al menos 3 dimensiones, recibido: {psi.dim()}")
    
    # Normalizar dimensiones
    if psi.dim() == 4 and psi.shape[0] == 1:
        psi = psi.squeeze(0)
    
    # Validar que psi tenga la última dimensión (d_state)
    if psi.shape[-1] == 0:
        raise ValueError("psi no puede tener d_state = 0")

    # Aplicar downsampling si se especifica (para optimizar transferencia de datos)
    if downsample_factor > 1:
        # Downsample usando promedio (pooling promedio)
        H, W = psi.shape[0], psi.shape[1]
        new_H, new_W = H // downsample_factor, W // downsample_factor
        if new_H > 0 and new_W > 0:
            # Reshape y promedio
            psi_downsampled = psi[:new_H * downsample_factor, :new_W * downsample_factor].reshape(
                new_H, downsample_factor, new_W, downsample_factor, -1
            ).mean(dim=(1, 3))
            psi = psi_downsampled
    
    # OPTIMIZACIÓN CUDA: Calcular en GPU primero, luego mover a CPU una sola vez
    # Esto evita múltiples sincronizaciones CUDA costosas
    device = psi.device if isinstance(psi, torch.Tensor) else torch.device('cpu')
    is_cuda = device.type == 'cuda'
    
    # OPTIMIZACIÓN: Usar inference_mode si está disponible (PyTorch 1.9+)
    # inference_mode es más rápido que no_grad() para operaciones que no necesitan grads
    use_inference_mode = hasattr(torch, 'inference_mode')
    
    if use_inference_mode:
        inference_context = torch.inference_mode()
    else:
        inference_context = torch.no_grad()
    
    with inference_context:  # Más rápido que no_grad() para inferencia pura
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
    
    # OPTIMIZACIÓN: Mover todos los datos a CPU en un solo paso (una sincronización)
    # Usar .detach() para evitar problemas con el grafo computacional
    # Mantener cálculos en GPU el mayor tiempo posible antes de convertir a numpy
    # Si estamos en CUDA, sincronizar una sola vez al final
    if is_cuda:
        # Asegurar que todos los cálculos GPU terminen antes de copiar
        torch.cuda.synchronize()
    
    # Mover todo a CPU en batch (una sola sincronización CUDA)
    # OPTIMIZACIÓN: Usar .contiguous() antes de .numpy() para mejor rendimiento
    # Verificar si ya son numpy arrays (puede pasar con motor nativo que ya convierte)
    # CRÍTICO: Usar try-except para manejar casos donde isinstance() puede fallar o objetos híbridos
    try:
        if isinstance(density, torch.Tensor) and hasattr(density, 'detach'):
            density = density.detach().contiguous().cpu().numpy()
        elif not isinstance(density, np.ndarray):
            density = np.array(density)
    except (AttributeError, TypeError) as e:
        logging.warning(f"Error convirtiendo density: {e}, intentando np.array()")
        try:
            density = np.array(density)
        except Exception as e2:
            logging.error(f"Error crítico convirtiendo density: {e2}")
            raise
    
    try:
        if isinstance(phase, torch.Tensor) and hasattr(phase, 'detach'):
            phase = phase.detach().contiguous().cpu().numpy()
        elif not isinstance(phase, np.ndarray):
            phase = np.array(phase)
    except (AttributeError, TypeError) as e:
        logging.warning(f"Error convirtiendo phase: {e}, intentando np.array()")
        try:
            phase = np.array(phase)
        except Exception as e2:
            logging.error(f"Error crítico convirtiendo phase: {e2}")
            raise
    
    try:
        if isinstance(real_part, torch.Tensor) and hasattr(real_part, 'detach'):
            real_part = real_part.detach().contiguous().cpu().numpy()
        elif not isinstance(real_part, np.ndarray):
            real_part = np.array(real_part)
    except (AttributeError, TypeError) as e:
        logging.warning(f"Error convirtiendo real_part: {e}, intentando np.array()")
        try:
            real_part = np.array(real_part)
        except Exception as e2:
            logging.error(f"Error crítico convirtiendo real_part: {e2}")
            raise
    
    try:
        if isinstance(imag_part, torch.Tensor) and hasattr(imag_part, 'detach'):
            imag_part = imag_part.detach().contiguous().cpu().numpy()
        elif not isinstance(imag_part, np.ndarray):
            imag_part = np.array(imag_part)
    except (AttributeError, TypeError) as e:
        logging.warning(f"Error convirtiendo imag_part: {e}, intentando np.array()")
        try:
            imag_part = np.array(imag_part)
        except Exception as e2:
            logging.error(f"Error crítico convirtiendo imag_part: {e2}")
            raise
    
    try:
        if isinstance(energy, torch.Tensor) and hasattr(energy, 'detach'):
            energy = energy.detach().contiguous().cpu().numpy()
        elif not isinstance(energy, np.ndarray):
            energy = np.array(energy)
    except (AttributeError, TypeError) as e:
        logging.warning(f"Error convirtiendo energy: {e}, intentando np.array()")
        try:
            energy = np.array(energy)
        except Exception as e2:
            logging.error(f"Error crítico convirtiendo energy: {e2}")
            raise
    
    # Calcular gradiente espacial (magnitud del gradiente)
    # OPTIMIZACIÓN: Calcular gradiente en GPU si es posible, luego mover a CPU
    if len(density.shape) == 2:
        # Convertir densidad a tensor si es necesario para cálculo en GPU
        density_tensor = torch.tensor(density) if not isinstance(density, torch.Tensor) else density
        if isinstance(density_tensor, torch.Tensor):
            # Calcular gradiente en GPU usando torch.diff
            grad_y_torch = torch.diff(density_tensor, dim=0, prepend=density_tensor[0:1, :])
            grad_x_torch = torch.diff(density_tensor, dim=1, prepend=density_tensor[:, 0:1])
            gradient_magnitude_tensor = torch.sqrt(grad_x_torch**2 + grad_y_torch**2)
            gradient_magnitude = gradient_magnitude_tensor.cpu().numpy()
        else:
            # Fallback a numpy si no es tensor
            grad_y, grad_x = np.gradient(density)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    else:
        gradient_magnitude = np.zeros_like(density)

    # --- Lógica de selección de datos del mapa ---
    map_data = None
    if viz_type == 'density':
        map_data = density
    elif viz_type == 'phase':
        # Normalizar fase a [0, 1] con mejor contraste
        map_data = (phase + np.pi) / (2 * np.pi)
        # Aplicar un estiramiento de contraste para mejor visibilidad
        p2, p98 = np.percentile(map_data, [2, 98])
        if p98 > p2:
            map_data = np.clip((map_data - p2) / (p98 - p2), 0, 1)
    elif viz_type == 'phase_hsv':
        # Fase en HSV: H = fase, S = 1, V = densidad normalizada
        phase_normalized = (phase + np.pi) / (2 * np.pi)  # [0, 1]
        density_norm = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-10)
        # Combinar fase (hue) con densidad (value) - se convertirá a RGB en el frontend
        map_data = phase_normalized  # El frontend usará esto como hue
        # Guardar datos HSV en result (se añadirá después de inicializar result)
    elif viz_type == 'energy':
        map_data = energy
    elif viz_type == 'real':
        # Parte real del primer canal o promedio
        if len(real_part.shape) == 3:  # (H, W, d_state)
            map_data = np.mean(real_part, axis=-1) if real_part.shape[-1] > 1 else real_part[:, :, 0]
        elif len(real_part.shape) == 2:
            map_data = real_part
        else:
            map_data = real_part.flatten().reshape(psi.shape[0], psi.shape[1]) if len(psi.shape) >= 2 else real_part
    elif viz_type == 'imag':
        # Parte imaginaria del primer canal o promedio
        if len(imag_part.shape) == 3:  # (H, W, d_state)
            map_data = np.mean(imag_part, axis=-1) if imag_part.shape[-1] > 1 else imag_part[:, :, 0]
        elif len(imag_part.shape) == 2:
            map_data = imag_part
        else:
            map_data = imag_part.flatten().reshape(psi.shape[0], psi.shape[1]) if len(psi.shape) >= 2 else imag_part
    elif viz_type == 'gradient':
        map_data = gradient_magnitude
    elif viz_type == 'spectral':
        # Transformada de Fourier (magnitud del espectro)
        if len(density.shape) == 2:
            fft = np.fft.fft2(density)
            fft_shifted = np.fft.fftshift(fft)
            spectral_magnitude = np.abs(fft_shifted)
            # Log scale para mejor visualización
            map_data = np.log1p(spectral_magnitude)
        else:
            map_data = density
    elif viz_type == 'physics':
        # Mapa de física: muestra la "fuerza" de la interacción local (matriz A)
        if motor is not None and hasattr(motor, 'get_physics_matrix_map'):
            physics_map = motor.get_physics_matrix_map()
            if physics_map is not None:
                map_data = physics_map
            else:
                # Fallback: usar densidad si no se puede calcular física
                map_data = density
        else:
            # Fallback: usar densidad si no hay motor
            map_data = density
    elif viz_type == 'entropy':
        # Mapa de entropía: mide la complejidad/información por célula
        # Entropía de Shannon: H = -Σ p_i * log(p_i)
        # Donde p_i = |psi[i]|² / Σ|psi|² (probabilidad de cada canal)
        try:
            psi_abs_sq = np.abs(psi.cpu().numpy())**2  # (H, W, d_state)
            # Normalizar para obtener probabilidades
            total_prob = np.sum(psi_abs_sq, axis=-1, keepdims=True)  # (H, W, 1)
            probabilities = np.where(total_prob > 1e-10, 
                                    psi_abs_sq / (total_prob + 1e-10),
                                    1.0 / psi.shape[-1])  # Distribución uniforme si total es 0
            # Calcular entropía
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=-1)  # (H, W)
            # Normalizar por entropía máxima (log(d_state))
            max_entropy = np.log(psi.shape[-1])
            map_data = entropy / (max_entropy + 1e-10)
        except Exception as e:
            logging.warning(f"Error calculando entropía: {e}")
            map_data = density
    elif viz_type == 'coherence':
        # Mapa de coherencia: mide la coherencia de fase entre células vecinas
        # Coherencia = |⟨ψ(x,y) | ψ(x+1,y)⟩| / (|ψ(x,y)| * |ψ(x+1,y)|)
        try:
            psi_np = psi.cpu().numpy()  # (H, W, d_state) complejo
            H, W, d_state = psi_np.shape
            
            # Calcular coherencia horizontal (entre vecinos en x)
            psi_shifted_x = np.roll(psi_np, shift=-1, axis=1)  # Desplazar en x
            psi_shifted_x[:, -1, :] = 0  # Borde: sin vecino
            
            # Producto interno: ⟨ψ | ψ_shifted⟩
            inner_product = np.sum(psi_np * np.conj(psi_shifted_x), axis=-1)  # (H, W)
            
            # Normas
            norm_psi = np.sqrt(np.sum(np.abs(psi_np)**2, axis=-1))  # (H, W)
            norm_shifted = np.sqrt(np.sum(np.abs(psi_shifted_x)**2, axis=-1))  # (H, W)
            
            # Coherencia
            coherence = np.abs(inner_product) / (norm_psi * norm_shifted + 1e-10)  # (H, W)
            map_data = coherence
        except Exception as e:
            logging.warning(f"Error calculando coherencia: {e}")
            map_data = density
    elif viz_type == 'channel_activity':
        # Actividad por canal: muestra qué canales están más activos
        # Visualizar como promedio de actividad por canal en cada posición
        try:
            psi_abs_sq = np.abs(psi.cpu().numpy())**2  # (H, W, d_state)
            H, W = psi_abs_sq.shape[0], psi_abs_sq.shape[1]
            # Calcular actividad promedio por canal (promedio espacial)
            channel_activity = np.mean(psi_abs_sq, axis=(0, 1))  # (d_state,)
            # Encontrar el canal más activo para cada célula
            dominant_channel = np.argmax(psi_abs_sq, axis=-1)  # (H, W)
            # Normalizar por actividad del canal dominante
            max_activity = np.max(channel_activity)
            if max_activity > 0:
                # Mapa: actividad del canal dominante en cada posición
                map_data = np.array([[psi_abs_sq[y, x, dominant_channel[y, x]] / max_activity 
                                     for x in range(W)] for y in range(H)])
            else:
                map_data = density
        except Exception as e:
            logging.warning(f"Error calculando actividad de canales: {e}")
            map_data = density
    else:
        map_data = density

    # Normalizar map_data a [0, 1]
    min_val, max_val = np.min(map_data), np.max(map_data)
    if max_val > min_val:
        map_data = (map_data - min_val) / (max_val - min_val)
    else:
        map_data = np.zeros_like(map_data)

    # --- Cálculo de datos para Poincaré (solo si se necesita) ---
    poincare_coords = [[0.0, 0.0]]  # Default
    if viz_type in ['poincare', 'poincare_3d']:
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
            else:
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
                else:
                    logging.warning(f"Poincaré: no hay suficientes puntos ({psi_flat_for_pca.shape[0]})")
        except Exception as e:
            logging.warning(f"Error al calcular coordenadas de Poincaré: {e}. Usando coordenadas por defecto.", exc_info=True)

    # --- Histogramas (solo si se necesitan) ---
    hist_data = {}
    if viz_type == 'histogram':
        density_flat = density.flatten()
        phase_flat = phase.flatten()
        real_flat = real_part.flatten()
        imag_flat = imag_part.flatten()

        density_hist, density_bins = np.histogram(density_flat, bins=30, range=(0, np.max(density_flat) if np.max(density_flat) > 0 else 1))
        phase_hist, phase_bins = np.histogram(phase_flat, bins=30, range=(-np.pi, np.pi))
        real_hist, real_bins = np.histogram(real_flat, bins=30, range=(-1, 1))
        imag_hist, imag_bins = np.histogram(imag_flat, bins=30, range=(-1, 1))

        hist_data = {
            'density': [{"bin": f"{density_bins[i]:.2f}", "count": int(density_hist[i])} for i in range(len(density_hist))],
            'phase': [{"bin": f"{phase_bins[i]:.2f}", "count": int(phase_hist[i])} for i in range(len(phase_hist))],
            'real': [{"bin": f"{real_bins[i]:.2f}", "count": int(real_hist[i])} for i in range(len(real_hist))],
            'imag': [{"bin": f"{imag_bins[i]:.2f}", "count": int(imag_hist[i])} for i in range(len(imag_hist))],
        }

    result = {
        "map_data": map_data.tolist(),
        "hist_data": hist_data,
        "poincare_coords": poincare_coords.tolist() if isinstance(poincare_coords, np.ndarray) else poincare_coords
    }
    
    # Datos para visualización 3D compleja (real vs imag vs tiempo)
    if viz_type == 'complex_3d':
        # Extraer parte real e imaginaria promediadas sobre canales
        if len(real_part.shape) == 3:  # (H, W, d_state)
            real_avg = np.mean(real_part, axis=-1) if real_part.shape[-1] > 1 else real_part[:, :, 0]
        else:
            real_avg = real_part
        if len(imag_part.shape) == 3:  # (H, W, d_state)
            imag_avg = np.mean(imag_part, axis=-1) if imag_part.shape[-1] > 1 else imag_part[:, :, 0]
        else:
            imag_avg = imag_part
        result["complex_3d_data"] = {
            "real": real_avg.tolist(),
            "imag": imag_avg.tolist()
        }
    
    # Añadir datos HSV si se calculó
    if viz_type == 'phase_hsv':
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        density_norm = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-10)
        result['phase_hsv_data'] = {
            'hue': phase_normalized.tolist(),
            'saturation': np.ones_like(phase_normalized).tolist(),
            'value': density_norm.tolist()
        }
    
    # --- NUEVAS VISUALIZACIONES ---
    
    # 1. Phase Attractor: Solo calcular si se necesita
    if viz_type == 'phase_attractor' and psi.shape[-1] >= 2:
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
            center_y, center_x = 0, 0
        
        # Extraer canales 0 y 1
        channel_0 = psi_center[0].cpu().numpy()
        channel_1 = psi_center[1].cpu().numpy() if psi.shape[-1] > 1 else psi_center[0].cpu().numpy()
        
        result["phase_attractor"] = {
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
    else:
        result["phase_attractor"] = None
    
    # 2. Flow Viewer: Solo calcular si se necesita (viz_type == 'flow')
    if delta_psi is not None and viz_type == 'flow':
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
                
                result["flow_data"] = {
                    "dx": dx_scaled.tolist(),
                    "dy": dy_scaled.tolist(),
                    "magnitude": magnitude_norm.tolist()
                }
            else:
                result["flow_data"] = None
        except Exception as e:
            logging.warning(f"Error calculando datos de flujo: {e}")
            result["flow_data"] = None
    else:
        result["flow_data"] = None
    
    return result