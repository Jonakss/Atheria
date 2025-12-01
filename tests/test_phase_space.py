import pytest
import torch
import numpy as np
from src.pipelines.viz.phase_space import get_phase_space_data

def test_get_phase_space_data_pca():
    # Create a dummy quantum state [H, W, d_state]
    H, W, d_state = 32, 32, 4
    psi = torch.randn(H, W, d_state, dtype=torch.complex64)
    
    # Test PCA method
    result = get_phase_space_data(psi, method='pca', n_clusters=3, subsample=1.0)
    
    assert result['method'] == 'PCA'
    assert 'points' in result
    assert 'centroids' in result
    assert 'metrics' in result
    assert len(result['points']) > 0
    
    # Check point structure
    point = result['points'][0]
    assert 'x' in point
    assert 'y' in point
    assert 'z' in point
    assert 'cluster' in point
    assert 'color' in point
    assert 'orig_x' in point
    assert 'orig_y' in point

def test_get_phase_space_data_umap_fallback():
    # Test UMAP method (might fallback to PCA if UMAP not installed in test env, 
    # but we want to ensure it runs without error)
    H, W, d_state = 32, 32, 4
    psi = torch.randn(H, W, d_state, dtype=torch.complex64)
    
    result = get_phase_space_data(psi, method='umap', n_clusters=3, subsample=0.1)
    
    assert result['method'] in ['UMAP', 'PCA']
    assert len(result['points']) > 0

def test_get_phase_space_data_empty():
    psi = torch.tensor([])
    result = get_phase_space_data(psi)
    assert len(result['points']) == 0
