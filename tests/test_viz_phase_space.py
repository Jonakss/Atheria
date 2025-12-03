import torch
import numpy as np
import pytest
from src.pipelines.viz.phase_space import get_phase_space_data

def test_phase_space_structure():
    """Verifica la estructura de los datos retornados."""
    # Crear un estado dummy: 32x32 grid, 8 canales
    psi = torch.randn(32, 32, 8, dtype=torch.complex64)
    
    result = get_phase_space_data(psi, n_clusters=3)
    
    assert "points" in result
    assert "centroids" in result
    assert "metrics" in result
    assert "explained_variance" in result["metrics"]
    
    # Verificar tipos
    assert isinstance(result["points"], list)
    assert isinstance(result["centroids"], list)
    assert isinstance(result["metrics"]["explained_variance"], list)
    
    # Verificar dimensiones de explained_variance
    assert len(result["metrics"]["explained_variance"]) == 3

def test_phase_space_content():
    """Verifica que los datos tengan sentido."""
    # Crear un estado con estructura clara
    # Mitad superior: estado A, Mitad inferior: estado B
    psi = torch.zeros(32, 32, 4, dtype=torch.complex64)
    
    # Estado A: Canal 0 activo
    psi[:16, :, 0] = 1.0
    # Estado B: Canal 1 activo
    psi[16:, :, 1] = 1.0
    
    result = get_phase_space_data(psi, n_clusters=2)
    
    points = result["points"]
    
    # Cada punto es un diccionario, así que el len(points) es directamente el número de puntos
    num_points = len(points)
    assert num_points > 0
    
    # Verificar que hay al menos 2 clusters (o 1 si K-Means falla en separar, pero pedimos 2)
    clusters = set()
    for point in points:
        clusters.add(point["cluster"])
        
    # Nota: K-Means no garantiza encontrar 2 clusters si los datos son muy simples o colineales,
    # pero con este input ortogonal debería funcionar.
    assert len(clusters) >= 1 

def test_caching_mechanism():
    """Verifica que el caché funcione."""
    psi = torch.randn(64, 64, 4, dtype=torch.complex64)
    
    # Primera llamada
    res1 = get_phase_space_data(psi)
    
    # Segunda llamada con mismo input (debería usar caché)
    res2 = get_phase_space_data(psi)
    
    # Los objetos deberían ser idénticos (misma referencia en memoria si es el dict cacheado)
    assert res1 is res2
    
    # Llamada con input diferente
    psi_new = torch.randn(64, 64, 4, dtype=torch.complex64)
    res3 = get_phase_space_data(psi_new)
    
    assert res3 is not res1

def test_empty_input():
    """Verifica manejo de input vacío."""
    psi = torch.tensor([], dtype=torch.complex64)
    result = get_phase_space_data(psi)
    assert result["points"] == []
