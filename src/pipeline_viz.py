# src/pipeline_viz.py
import torch
import numpy as np
from sklearn.decomposition import PCA

# PCA global - se reutiliza entre frames para mejor rendimiento
pca = PCA(n_components=2)
# Cache para PCA - solo recalcular cada N frames
_pca_cache = {'last_step': -1, 'coords': None, 'step_interval': 10}  # Recalcular PCA cada 10 frames

def get_visualization_data(psi: torch.Tensor, viz_type: str, delta_psi: torch.Tensor = None, motor=None, downsample_factor: int = 1, current_step: int = 0, compute_histograms: bool = True, compute_poincare: bool = True):
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
    
    # OPTIMIZACIÓN: Calcular densidad en GPU y mover a CPU una sola vez
    density_tensor = torch.sum(psi.abs()**2, dim=-1)
    density = density_tensor.cpu().numpy()
    
    # OPTIMIZACIÓN: Calcular fase y partes real/imag en GPU antes de mover a CPU
    # Mejorar visualización de fase: usar el canal 0 o promedio ponderado
    if psi.shape[-1] > 0:
        # Calcular fase en GPU
        phase_weighted_tensor = torch.angle(psi)
        # Calcular promedio circular de fases en GPU (más eficiente)
        phase_cos = torch.cos(phase_weighted_tensor).mean(dim=-1)
        phase_sin = torch.sin(phase_weighted_tensor).mean(dim=-1)
        phase_tensor = torch.atan2(phase_sin, phase_cos)
        phase = phase_tensor.cpu().numpy()
    else:
        phase_tensor = torch.angle(psi)
        if phase_tensor.ndim > 2:
            phase_tensor = phase_tensor[..., 0]
        phase = phase_tensor.cpu().numpy()
    
    # OPTIMIZACIÓN: Mover real e imag a CPU una sola vez
    real_part = psi.real.cpu().numpy()
    imag_part = psi.imag.cpu().numpy()
    
    # OPTIMIZACIÓN: Reutilizar density_tensor para energía (ya calculado arriba)
    # energy es igual a density, así que reutilizamos
    energy = density
    
    # Calcular gradiente espacial (magnitud del gradiente)
    if len(density.shape) == 2:
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
            import logging
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
            import logging
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
            import logging
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

    # --- Cálculo de datos para Poincaré (OPTIMIZADO: solo cada N frames) ---
    global _pca_cache
    poincare_coords = None
    
    if compute_poincare:
        # OPTIMIZACIÓN: Solo calcular PCA cada N frames (muy costoso)
        should_recompute_pca = (
            current_step % _pca_cache['step_interval'] == 0 or 
            _pca_cache['coords'] is None
        )
        
        if should_recompute_pca:
            try:
                # OPTIMIZACIÓN: Usar datos ya en CPU si están disponibles
                psi_flat_real = psi.real.reshape(-1, psi.shape[-1]).cpu().numpy()
                psi_flat_imag = psi.imag.reshape(-1, psi.shape[-1]).cpu().numpy()
                psi_flat_for_pca = np.concatenate([psi_flat_real, psi_flat_imag], axis=1)
                
                # Validar que hay suficientes puntos para PCA
                if psi_flat_for_pca.shape[0] < 2:
                    poincare_coords = [[0.0, 0.0]]  # Coordenada por defecto
                else:
                    # OPTIMIZACIÓN: Usar partial_fit si es posible para mejor rendimiento
                    poincare_coords = pca.fit_transform(psi_flat_for_pca)
                    max_abs_val = np.max(np.abs(poincare_coords))
                    if max_abs_val > 0:
                        poincare_coords = poincare_coords / max_abs_val
                    
                    # Cachear resultado
                    _pca_cache['coords'] = poincare_coords
                    _pca_cache['last_step'] = current_step
            except Exception as e:
                import logging
                logging.warning(f"Error al calcular coordenadas de Poincaré: {e}. Usando coordenadas por defecto.")
                poincare_coords = [[0.0, 0.0]]
                _pca_cache['coords'] = poincare_coords
        else:
            # Reutilizar coordenadas cacheadas
            poincare_coords = _pca_cache['coords']
    else:
        poincare_coords = [[0.0, 0.0]]

    # --- Histogramas (OPTIMIZADO: solo calcular si se solicitan) ---
    # OPTIMIZACIÓN: Histogramas son costosos, calcular solo si se necesitan
    if compute_histograms:
        density_flat = density.flatten()
        phase_flat = phase.flatten()
        real_flat = real_part.flatten()
        imag_flat = imag_part.flatten()

        # OPTIMIZACIÓN: Usar menos bins si el grid es grande (reduce costo computacional)
        num_bins = 20 if density.size > 10000 else 30  # Menos bins para grids grandes
        
        density_hist, density_bins = np.histogram(density_flat, bins=num_bins, range=(0, np.max(density_flat) if np.max(density_flat) > 0 else 1))
        phase_hist, phase_bins = np.histogram(phase_flat, bins=num_bins, range=(-np.pi, np.pi))
        real_hist, real_bins = np.histogram(real_flat, bins=num_bins, range=(-1, 1))
        imag_hist, imag_bins = np.histogram(imag_flat, bins=num_bins, range=(-1, 1))

        hist_data = {
            'density': [{"bin": f"{density_bins[i]:.2f}", "count": int(density_hist[i])} for i in range(len(density_hist))],
            'phase': [{"bin": f"{phase_bins[i]:.2f}", "count": int(phase_hist[i])} for i in range(len(phase_hist))],
            'real': [{"bin": f"{real_bins[i]:.2f}", "count": int(real_hist[i])} for i in range(len(real_hist))],
            'imag': [{"bin": f"{imag_bins[i]:.2f}", "count": int(imag_hist[i])} for i in range(len(imag_hist))],
        }
    else:
        # Histogramas vacíos si no se calculan
        hist_data = {
            'density': [],
            'phase': [],
            'real': [],
            'imag': []
        }

    # OPTIMIZACIÓN: Serializar map_data de forma más eficiente
    # Para arrays grandes, usar .tolist() solo si es necesario
    # Para arrays pequeños/medianos, .tolist() es aceptable
    if map_data.size > 10000:
        # Para arrays grandes, convertir a lista de forma más eficiente
        # o usar compresión en el servidor
        map_data_list = map_data.tolist()
    else:
        map_data_list = map_data.tolist()
    
    result = {
        "map_data": map_data_list,
        "hist_data": hist_data,
        "poincare_coords": poincare_coords.tolist() if poincare_coords is not None else [[0.0, 0.0]]
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
    
    # 1. Phase Attractor: Scatter plot de canales 0 y 1 para célula central
    # La célula central se selecciona siempre en el centro del grid, independientemente del tamaño
    if psi.shape[-1] >= 2:
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
    
    # 2. Flow Viewer: Quiver plot con delta_psi
    # Calcular siempre flow_data si delta_psi está disponible (no solo cuando viz_type == 'flow')
    if delta_psi is not None:
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
            import logging
            logging.warning(f"Error calculando datos de flujo: {e}")
            result["flow_data"] = None
    else:
        result["flow_data"] = None
    
    return result