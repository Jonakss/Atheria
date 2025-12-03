import torch
import torch.nn as nn
import pytest
from src.engines.holographic_engine import HolographicEngine

class MockModel(nn.Module):
    def __init__(self, d_state):
        super().__init__()
        self.d_state = d_state
        
    def forward(self, x):
        # Identity op
        return x

def test_holographic_engine_initialization():
    grid_size = 64
    d_state = 4
    device = torch.device('cpu')
    model = MockModel(d_state)
    
    engine = HolographicEngine(model, grid_size, d_state, device, bulk_depth=8)
    
    assert engine.grid_size == grid_size
    assert engine.d_state == d_state
    assert engine.bulk_depth == 8
    assert engine.state.psi is not None

def test_holographic_projection():
    grid_size = 64
    d_state = 4
    device = torch.device('cpu')
    model = MockModel(d_state)
    
    engine = HolographicEngine(model, grid_size, d_state, device, bulk_depth=8)
    
    # Initialize with some random state
    engine.state.psi = torch.randn(1, grid_size, grid_size, d_state, dtype=torch.complex64)
    
    # Get bulk state
    bulk = engine.get_bulk_state()
    
    # Check shape: [1, D, H, W]
    assert bulk.shape == (1, 8, grid_size, grid_size)
    assert not torch.isnan(bulk).any()
    
    # Check that deeper layers are blurrier (lower variance)
    # Variance of layer 0 (boundary) should be higher than layer 7 (deep bulk)
    var_0 = torch.var(bulk[:, 0])
    var_7 = torch.var(bulk[:, 7])
    
    print(f"Variance Layer 0: {var_0}")
    print(f"Variance Layer 7: {var_7}")
    
    assert var_0 > var_7

if __name__ == "__main__":
    test_holographic_engine_initialization()
    test_holographic_projection()
    print("All tests passed!")
