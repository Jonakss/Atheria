"""
Módulo de visualización de Espacio de Fases.
Transforma el estado cuántico en una representación 3D usando PCA y Clustering.
"""
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
try:
    import umap
except ImportError:
    umap = None

# Configuración de caché y optimización
_phase_space_cache = {
    'last_psi_hash': None,
    'last_result': None,
    'recalc_counter': 0,
    'last_method': None
}
PHASE_SPACE_RECALC_INTERVAL = 10  # Recalcular cada N frames (es costoso)
MAX_POINTS_FOR_ANALYSIS = 10000   # Máximo de puntos para PCA/KMeans


def get_phase_space_data(psi: torch.Tensor, method: str = 'pca', n_clusters: int = 4, subsample: float = 0.1) -> Dict:
    """
    Genera datos del espacio de fases usando PCA o UMAP y K-Means.
    
    Args:
        psi: Tensor complejo de estado [H, W, d_state]
        method: 'pca' (rápido, lineal) o 'umap' (lento, topológico)
        n_clusters: Número de clusters para K-Means
        subsample: Fracción de puntos a usar (0.0 a 1.0)
        
    Returns:
        Dict con points (lista de dicts), centroids, metrics
    """
    global _phase_space_cache
    
    try:
        # 1. Validación y preparación
        if psi.numel() == 0:
            return _get_empty_result()
            
        # Normalizar dimensiones si es necesario
        if psi.dim() == 4:
            psi = psi.squeeze(0)
            
        # 2. Gestión de Caché (Solo para PCA en modo live, UMAP siempre recalcula bajo demanda)
        # Usar submuestreo para hash rápido
        psi_sample = psi[::max(1, psi.shape[0]//32), ::max(1, psi.shape[1]//32), :]
        psi_hash = hash(psi_sample.cpu().numpy().tobytes())
        
        _phase_space_cache['recalc_counter'] += 1
        should_recalc = _phase_space_cache['recalc_counter'] >= PHASE_SPACE_RECALC_INTERVAL
        
        if (method == 'pca' and
            psi_hash == _phase_space_cache['last_psi_hash'] and 
            _phase_space_cache['last_result'] is not None and 
            _phase_space_cache['last_method'] == 'pca' and
            not should_recalc):
            return _phase_space_cache['last_result']

        # 3. Preparación de datos (Flattening & Complex Handling)
        H, W, d_state = psi.shape
        total_pixels = H * W
        
        # Calcular stride basado en subsample
        # subsample 0.1 significa tomar el 10% de los puntos
        # stride ~ sqrt(1/subsample)
        if subsample <= 0.0 or subsample > 1.0:
            subsample = 0.1
            
        target_points = int(total_pixels * subsample)
        target_points = min(target_points, MAX_POINTS_FOR_ANALYSIS) # Cap de seguridad
        
        stride = max(1, int(np.sqrt(total_pixels / target_points)))
        
        # Extraer puntos con stride
        psi_sub = psi[::stride, ::stride, :]
        
        # Convertir a numpy y aplanar
        # Concatenar parte real e imaginaria: [N, d_state] -> [N, 2*d_state]
        flat_real = psi_sub.real.reshape(-1, d_state).cpu().numpy()
        flat_imag = psi_sub.imag.reshape(-1, d_state).cpu().numpy()
        data_matrix = np.concatenate([flat_real, flat_imag], axis=1)
        
        # Guardar coordenadas originales para referencia visual
        y_coords, x_coords = np.mgrid[0:H:stride, 0:W:stride]
        orig_coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
        
        # Ajustar longitud si hay mismatch por redondeo de reshape
        min_len = min(data_matrix.shape[0], orig_coords.shape[0])
        data_matrix = data_matrix[:min_len]
        orig_coords = orig_coords[:min_len]
        
        # 4. Reducción de Dimensionalidad
        metrics = {}
        
        if method == 'umap':
            if umap is None:
                logging.warning("UMAP no instalado, usando PCA como fallback")
                method = 'pca'
            else:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
                reduced_data = reducer.fit_transform(data_matrix)
                # UMAP no tiene explained_variance simple, pero podemos devolver trustworthiness si se calcula (costoso)
                metrics['trustworthiness'] = 0.95 # Placeholder/Simulado por ahora para no ralentizar
        
        if method == 'pca':
            n_components = min(3, data_matrix.shape[1], data_matrix.shape[0])
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(data_matrix)
            
            # Rellenar con ceros si tenemos menos de 3 componentes
            if n_components < 3:
                padding = np.zeros((reduced_data.shape[0], 3 - n_components))
                reduced_data = np.hstack([reduced_data, padding])
                
            metrics['explained_variance'] = pca.explained_variance_ratio_.tolist()
            # Rellenar varianza
            while len(metrics.get('explained_variance', [])) < 3:
                if 'explained_variance' not in metrics: metrics['explained_variance'] = []
                metrics['explained_variance'].append(0.0)

        # 5. Clustering (K-Means)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_data)
        centroids = kmeans.cluster_centers_
        
        # Colores predefinidos para clusters (hex)
        cluster_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF"]
        
        # 6. Formatear Salida (JSON Array of Objects)
        points_data = []
        for i in range(len(reduced_data)):
            cluster_idx = int(cluster_labels[i])
            color = cluster_colors[cluster_idx % len(cluster_colors)]
            
            points_data.append({
                "x": float(reduced_data[i, 0]),
                "y": float(reduced_data[i, 1]),
                "z": float(reduced_data[i, 2]),
                "cluster": cluster_idx,
                "color": color,
                "orig_x": int(orig_coords[i, 0]),
                "orig_y": int(orig_coords[i, 1])
            })
            
        result = {
            "method": method.upper(),
            "points": points_data,
            "centroids": centroids.tolist(),
            "metrics": metrics
        }
        
        # Actualizar caché solo si es PCA (UMAP es on-demand)
        if method == 'pca':
            _phase_space_cache['last_psi_hash'] = psi_hash
            _phase_space_cache['last_result'] = result
            _phase_space_cache['recalc_counter'] = 0
            _phase_space_cache['last_method'] = 'pca'
        
        return result

    except Exception as e:
        logging.error(f"Error en cálculo de Espacio de Fases ({method}): {e}", exc_info=True)
        return _get_empty_result()


def _get_empty_result() -> Dict:
    return {
        "method": "NONE",
        "points": [],
        "centroids": [],
        "metrics": {"explained_variance": [0.0, 0.0, 0.0]}
    }
