# src/analysis.py
"""
Módulo de análisis científico usando t-SNE para visualizar la estructura del universo simulado.

Implementa dos métodos clave:
1. Atlas del Universo: Análisis de evolución temporal (múltiples snapshots)
2. Mapa Químico: Análisis de tipos de células (un snapshot)
"""
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

def compress_snapshot(psi, compression_dim=64, method='pca'):
    """
    Comprime un snapshot del estado cuántico a un vector de dimensión menor.
    
    Args:
        psi: Tensor de forma [1, H, W, d_state] o [H, W, d_state]
        compression_dim: Dimensión objetivo del vector comprimido
        method: Método de compresión ('pca' o 'mean')
    
    Returns:
        Vector comprimido de forma [compression_dim]
    """
    # Normalizar dimensiones
    if isinstance(psi, torch.Tensor):
        psi_np = psi.cpu().numpy()
    else:
        psi_np = psi
    
    # Manejar tensores complejos: convertir a formato real concatenando real e imag
    if np.iscomplexobj(psi_np):
        # Si es complejo, concatenar real e imag
        psi_real = psi_np.real
        psi_imag = psi_np.imag
        psi_np = np.concatenate([psi_real, psi_imag], axis=-1)  # Duplicar última dimensión
    
    # Aplanar a [H*W, d_state] o [H*W*d_state]
    if len(psi_np.shape) == 4 and psi_np.shape[0] == 1:
        psi_np = psi_np.squeeze(0)
    
    if len(psi_np.shape) == 3:  # [H, W, d_state]
        H, W, d_state = psi_np.shape
        psi_flat = psi_np.reshape(-1, d_state)  # [H*W, d_state]
    else:
        psi_flat = psi_np.flatten().reshape(-1, 1)
    
    if method == 'pca':
        # Usar PCA para comprimir
        if psi_flat.shape[0] < compression_dim:
            # Si hay menos muestras que dimensiones objetivo, usar todas las dimensiones
            pca = PCA(n_components=min(psi_flat.shape[0], psi_flat.shape[1]))
        else:
            pca = PCA(n_components=compression_dim)
        
        # Aplicar PCA sobre las células (cada célula es un punto)
        psi_compressed = pca.fit_transform(psi_flat)  # [H*W, compression_dim]
        # Promediar sobre todas las células para obtener un vector único
        compressed_vector = psi_compressed.mean(axis=0)  # [compression_dim]
        
        # Si el vector resultante es más pequeño que compression_dim, rellenar con ceros
        if len(compressed_vector) < compression_dim:
            padding = np.zeros(compression_dim - len(compressed_vector))
            compressed_vector = np.concatenate([compressed_vector, padding])
        
        return compressed_vector[:compression_dim]
    
    elif method == 'mean':
        # Método simple: estadísticas agregadas
        # Calcular estadísticas por canal
        mean_per_channel = psi_flat.mean(axis=0)  # [d_state]
        std_per_channel = psi_flat.std(axis=0)  # [d_state]
        
        # Concatenar estadísticas (ya están en formato real después de la conversión)
        stats = np.concatenate([mean_per_channel, std_per_channel])
        
        # Ajustar tamaño
        if len(stats) > compression_dim:
            return stats[:compression_dim]
        else:
            padding = np.zeros(compression_dim - len(stats))
            return np.concatenate([stats, padding])
    
    else:
        raise ValueError(f"Método de compresión desconocido: {method}")


def analyze_universe_atlas(psi_snapshots, compression_dim=64, perplexity=30, n_iter=1000):
    """
    Crea un "Atlas del Universo" analizando la evolución temporal.
    
    Args:
        psi_snapshots: Lista de tensores psi, cada uno es un snapshot en el tiempo
        compression_dim: Dimensión para comprimir cada snapshot
        perplexity: Parámetro de t-SNE (controla el balance local/global)
        n_iter: Número de iteraciones de t-SNE
    
    Returns:
        dict con:
            - coords: Array de forma [n_snapshots, 2] con coordenadas 2D de t-SNE
            - compressed_vectors: Array de forma [n_snapshots, compression_dim]
            - timesteps: Lista de pasos de tiempo correspondientes
    """
    if not psi_snapshots or len(psi_snapshots) == 0:
        raise ValueError("La lista de snapshots está vacía")
    
    logging.info(f"Analizando Atlas del Universo con {len(psi_snapshots)} snapshots...")
    
    # 1. Comprimir cada snapshot a un vector
    compressed_vectors = []
    for i, psi in enumerate(psi_snapshots):
        try:
            compressed = compress_snapshot(psi, compression_dim=compression_dim)
            compressed_vectors.append(compressed)
        except Exception as e:
            logging.warning(f"Error comprimiendo snapshot {i}: {e}")
            continue
    
    if len(compressed_vectors) < 2:
        raise ValueError(f"Se necesitan al menos 2 snapshots válidos, se obtuvieron {len(compressed_vectors)}")
    
    compressed_vectors = np.array(compressed_vectors)  # [n_snapshots, compression_dim]
    
    # 2. Normalizar los vectores comprimidos
    scaler = StandardScaler()
    compressed_normalized = scaler.fit_transform(compressed_vectors)
    
    # 3. Aplicar t-SNE
    # Ajustar perplexity si hay muy pocos snapshots
    actual_perplexity = min(perplexity, len(compressed_normalized) - 1)
    
    tsne = TSNE(n_components=2, perplexity=actual_perplexity, n_iter=n_iter, 
                random_state=42, verbose=0)
    coords_2d = tsne.fit_transform(compressed_normalized)  # [n_snapshots, 2]
    
    logging.info(f"Atlas del Universo completado. {len(coords_2d)} puntos proyectados en 2D.")
    
    return {
        'coords': coords_2d.tolist(),
        'compressed_vectors': compressed_vectors.tolist(),
        'timesteps': list(range(len(coords_2d)))
    }


def analyze_cell_chemistry(psi, n_samples=10000, perplexity=30, n_iter=1000):
    """
    Crea un "Mapa Químico" analizando los tipos de células en un snapshot.
    
    Args:
        psi: Tensor de forma [1, H, W, d_state] o [H, W, d_state]
        n_samples: Número máximo de células a muestrear (para eficiencia)
        perplexity: Parámetro de t-SNE
        n_iter: Número de iteraciones de t-SNE
    
    Returns:
        dict con:
            - coords: Array de forma [n_cells, 2] con coordenadas 2D de t-SNE
            - cell_indices: Lista de índices (y, x) de las células analizadas
            - cell_vectors: Array de forma [n_cells, d_state*2] con los vectores originales
    """
    if psi is None:
        raise ValueError("psi no puede ser None")
    
    # Normalizar dimensiones
    if isinstance(psi, torch.Tensor):
        psi_np = psi.cpu().numpy()
    else:
        psi_np = psi
    
    if len(psi_np.shape) == 4 and psi_np.shape[0] == 1:
        psi_np = psi_np.squeeze(0)
    
    if len(psi_np.shape) != 3:  # Debe ser [H, W, d_state]
        raise ValueError(f"psi debe tener forma [H, W, d_state], recibido: {psi_np.shape}")
    
    H, W, d_state = psi_np.shape
    total_cells = H * W
    
    logging.info(f"Analizando Mapa Químico: {total_cells} células, {d_state} canales por célula...")
    
    # 1. Extraer vectores de células: [H, W, d_state] -> [H*W, d_state]
    # Convertir a formato real concatenando real e imag
    psi_real = psi_np.real  # [H, W, d_state]
    psi_imag = psi_np.imag  # [H, W, d_state]
    
    # Aplanar y concatenar
    cells_real = psi_real.reshape(H * W, d_state)  # [H*W, d_state]
    cells_imag = psi_imag.reshape(H * W, d_state)  # [H*W, d_state]
    cell_vectors = np.concatenate([cells_real, cells_imag], axis=1)  # [H*W, d_state*2]
    
    # 2. Muestrear células si hay demasiadas (para eficiencia)
    if total_cells > n_samples:
        indices = np.random.choice(total_cells, n_samples, replace=False)
        cell_vectors_sampled = cell_vectors[indices]
        
        # Calcular índices (y, x) originales
        cell_indices = [(idx // W, idx % W) for idx in indices]
    else:
        cell_vectors_sampled = cell_vectors
        cell_indices = [(i // W, i % W) for i in range(total_cells)]
    
    n_cells = len(cell_vectors_sampled)
    logging.info(f"Analizando {n_cells} células...")
    
    # 3. Normalizar los vectores de células
    scaler = StandardScaler()
    cell_vectors_normalized = scaler.fit_transform(cell_vectors_sampled)
    
    # 4. Aplicar t-SNE
    # Ajustar perplexity si hay muy pocas células
    actual_perplexity = min(perplexity, n_cells - 1)
    
    tsne = TSNE(n_components=2, perplexity=actual_perplexity, n_iter=n_iter,
                random_state=42, verbose=0)
    coords_2d = tsne.fit_transform(cell_vectors_normalized)  # [n_cells, 2]
    
    logging.info(f"Mapa Químico completado. {len(coords_2d)} células proyectadas en 2D.")
    
    # Convertir índices a listas de enteros nativos de Python (no numpy int64)
    cell_indices_python = [[int(y), int(x)] for y, x in cell_indices]
    
    return {
        'coords': coords_2d.tolist(),
        'cell_indices': cell_indices_python,
        'cell_vectors': cell_vectors_sampled.tolist()
    }


def calculate_phase_map_metrics(tsne_coords):
    """
    Calcula métricas útiles para interpretar el mapa de fases.
    
    Args:
        tsne_coords: Array de forma [n_points, 2] con coordenadas t-SNE
    
    Returns:
        dict con métricas:
            - n_clusters: Estimación del número de clusters
            - spread: Dispersión de los puntos
            - density: Densidad promedio
    """
    coords = np.array(tsne_coords)
    
    # Calcular dispersión (distancia promedio al centroide)
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    spread = distances.mean()
    
    # Calcular densidad aproximada (puntos por unidad de área)
    # Área aproximada usando bounding box
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    area = x_range * y_range if x_range > 0 and y_range > 0 else 1.0
    density = len(coords) / area
    
    return {
        'spread': float(spread),
        'density': float(density),
        'n_points': int(len(coords))  # Convertir a int nativo de Python
    }

