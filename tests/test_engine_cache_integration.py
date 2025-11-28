import pytest
import torch
import torch.nn as nn
import numpy as np
from src.engines.qca_engine import Aetheria_Motor
from src.cache import cache
import time
import os

# Mock config
class MockConfig:
    CACHE_STATE_INTERVAL = 1
    CACHE_TTL = 60
    EXPERIMENT_NAME = "test_integration"
    INITIAL_STATE_MODE_INFERENCE = "random"

@pytest.fixture
def mock_motor():
    # Force cache reconnection
    from src.cache.dragonfly_client import DragonflyCache
    DragonflyCache._instance = None
    cache_instance = DragonflyCache(host='localhost', port=6379)
    
    # Update global cache reference if needed (though singleton should handle it)
    import src.cache
    src.cache.cache = cache_instance
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Simple identity model that preserves shape
    # Input is concatenated real+imag (16+16=32 channels)
    # Output should also be 32 channels (real+imag)
    model = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    
    motor = Aetheria_Motor(
        model_operator=model,
        grid_size=64,
        d_state=16,
        device=torch.device(device),
        cfg=MockConfig()
    )
    return motor

def test_engine_caches_state(mock_motor):
    # Ensure cache is enabled and clear previous test keys
    if not cache.enabled:
        pytest.skip("Dragonfly cache not available")
    
    cache.clear_pattern("state:test_integration:*")
    
    # Run step 1
    mock_motor.evolve_internal_state(step=1)
    
    # Check if key exists
    key = "state:test_integration:1"
    assert cache.exists(key)
    
    # Verify content
    cached_data = cache.get(key)
    assert cached_data is not None
    assert cached_data.shape == (1, 64, 64, 16)
    
    # Verify it matches current state
    current_state = mock_motor.state.psi.cpu().numpy()
    assert np.allclose(cached_data, current_state, atol=1e-5)

def test_engine_restores_state(mock_motor):
    if not cache.enabled:
        pytest.skip("Dragonfly cache not available")
        
    # Create a dummy state in cache for step 10
    dummy_state = np.random.randn(1, 64, 64, 16).astype(np.complex64)
    key = "state:test_integration:10"
    cache.set(key, dummy_state)
    
    # Run step 10 - should hit cache
    mock_motor.evolve_internal_state(step=10)
    
    # Verify motor state matches cached state
    current_state = mock_motor.state.psi.cpu().numpy()
    assert np.allclose(current_state, dummy_state, atol=1e-5)
    
    # Clean up
    cache.delete(key)
