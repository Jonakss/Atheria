import pytest
import torch
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.physics.quantum_collapse import IonQCollapse
from src.physics.steering import QuantumSteering

@pytest.fixture
def mock_device():
    return torch.device('cpu')

@pytest.fixture
def dummy_state(mock_device):
    # [1, H, W, d_state]
    return torch.zeros(1, 16, 16, 4, device=mock_device, dtype=torch.complex64)

def test_ionq_collapse_mock(mock_device, dummy_state):
    """Test that IonQCollapse works in mock mode when no backend is available."""
    collapser = IonQCollapse(mock_device)
    # Force mock just in case env vars are set
    collapser.backend = None
    
    # Apply collapse
    new_state = collapser.collapse(dummy_state, intensity=0.5)
    
    assert new_state.shape == dummy_state.shape
    assert not torch.equal(new_state, dummy_state) # Should have changed (noise added)
    assert new_state.dtype == dummy_state.dtype

def test_steering_inject_vortex(mock_device, dummy_state):
    """Test injecting a vortex pattern."""
    steering = QuantumSteering(mock_device)
    
    # Inject vortex at center
    new_state = steering.inject(dummy_state, 'vortex', x=8, y=8, radius=4)
    
    assert new_state.shape == dummy_state.shape
    # Check that center is not zero anymore
    center_val = new_state[0, 8, 8, :].abs().sum()
    assert center_val > 0

def test_steering_inject_plane_wave(mock_device, dummy_state):
    """Test injecting a plane wave."""
    steering = QuantumSteering(mock_device)
    
    new_state = steering.inject(dummy_state, 'plane_wave', x=0, y=0) # x,y ignored for global wave usually
    
    assert new_state.shape == dummy_state.shape
    assert new_state.abs().sum() > 0
