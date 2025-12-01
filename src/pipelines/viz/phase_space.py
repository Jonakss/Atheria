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

# Configuración de caché y optimización
_phase_space_cache = {
    'last_psi_hash': None,
    'last_result': None,
    'recalc_counter': 0
}
PHASE_SPACE_RECALC_INTERVAL = 10  # Recalcular cada N frames (es costoso)
MAX_POINTS_FOR_ANALYSIS = 10000   # Máximo de puntos para PCA/KMeans


def get_phase_space_data(psi: torch.Tensor, n_clusters: int = 3) -> Dict:
    """
    Genera datos del espacio de fases usando PCA y K-Means.
    
    Args:
        psi: Tensor complejo de estado [H, W, d_state]
        n_clusters: Número de clusters para K-Means
        
    Returns:
        Dict con points, centroids, explained_variance
    """
    global _phase_space_cache
    
    try:
        # 1. Validación y preparación
        if psi.numel() == 0:
            return _get_empty_result()
            
        # Normalizar dimensiones si es necesario
        if psi.dim() == 4:
            psi = psi.squeeze(0)
            
        # 2. Gestión de Caché
        # Usar submuestreo para hash rápido
        psi_sample = psi[::max(1, psi.shape[0]//32), ::max(1, psi.shape[1]//32), :]
        psi_hash = hash(psi_sample.cpu().numpy().tobytes())
        
        _phase_space_cache['recalc_counter'] += 1
        should_recalc = _phase_space_cache['recalc_counter'] >= PHASE_SPACE_RECALC_INTERVAL
        
        if (psi_hash == _phase_space_cache['last_psi_hash'] and 
            _phase_space_cache['last_result'] is not None and 
            not should_recalc):
            return _phase_space_cache['last_result']

        # 3. Preparación de datos (Flattening & Complex Handling)
        # Submuestreo inteligente para mantener rendimiento
        H, W, d_state = psi.shape
        total_pixels = H * W
        
        # Calcular stride para no exceder MAX_POINTS_FOR_ANALYSIS
        stride = max(1, int(np.sqrt(total_pixels / MAX_POINTS_FOR_ANALYSIS)))
        
        # Extraer puntos con stride
        psi_sub = psi[::stride, ::stride, :]
        
        # Convertir a numpy y aplanar
        # Concatenar parte real e imaginaria: [N, d_state] -> [N, 2*d_state]
        flat_real = psi_sub.real.reshape(-1, d_state).cpu().numpy()
        flat_imag = psi_sub.imag.reshape(-1, d_state).cpu().numpy()
        data_matrix = np.concatenate([flat_real, flat_imag], axis=1)
        
        # Guardar coordenadas originales para referencia visual
        # Crear grid de coordenadas
        y_coords, x_coords = np.mgrid[0:H:stride, 0:W:stride]
        orig_coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
        
        # 4. Reducción de Dimensionalidad (PCA)
        # Reducir a 3 componentes principales
        n_components = min(3, data_matrix.shape[1], data_matrix.shape[0])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_matrix)
        
        # Rellenar con ceros si tenemos menos de 3 componentes
        if n_components < 3:
            padding = np.zeros((pca_result.shape[0], 3 - n_components))
            pca_result = np.hstack([pca_result, padding])
            
        explained_variance = pca.explained_variance_ratio_.tolist()
        # Rellenar varianza si es necesario
        while len(explained_variance) < 3:
            explained_variance.append(0.0)

        # 5. Clustering (K-Means)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_result)
        centroids = kmeans.cluster_centers_
        
        # 6. Formatear Salida
        # Estructura optimizada: lista plana [x, y, z, cluster, orig_x, orig_y]
        # Esto es más eficiente para transferir y parsear en JS
        points_data = []
        for i in range(len(pca_result)):
            points_data.extend([
                float(pca_result[i, 0]),
                float(pca_result[i, 1]),
                float(pca_result[i, 2]),
                int(cluster_labels[i]),
                int(orig_coords[i, 0]),
                int(orig_coords[i, 1])
            ])
            
        result = {
            "points": points_data,
            "centroids": centroids.tolist(),
            "explained_variance": explained_variance
        }
        
        # Actualizar caché
        _phase_space_cache['last_psi_hash'] = psi_hash
        _phase_space_cache['last_result'] = result
        _phase_space_cache['recalc_counter'] = 0
        
        return result

    except Exception as e:
        logging.error(f"Error en cálculo de Espacio de Fases: {e}", exc_info=True)
        return _get_empty_result()


def _get_empty_result() -> Dict:
    return {
        "points": [],
        "centroids": [],
        "explained_variance": [0.0, 0.0, 0.0]
    }
