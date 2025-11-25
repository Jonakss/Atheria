import torch
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.engines.harmonic_engine import SparseHarmonicEngine

class MockModel(torch.nn.Module):
    def __init__(self, d_state):
        super().__init__()
        self.conv = torch.nn.Conv2d(d_state, d_state, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Simple diffusion + persistence
        return x + 0.1

def test_harmonic_engine():
    print("🧪 Testing SparseHarmonicEngine...")
    
    device = "cpu"
    d_state = 8
    model = MockModel(d_state).to(device)
    
    engine = SparseHarmonicEngine(model, d_state, device)
    
    # 1. Add initial matter
    print("   Adding initial matter...")
    initial_state = torch.randn(d_state, device=device)
    engine.add_matter(0, 0, 0, initial_state)
    
    assert len(engine.matter) == 1
    assert (0, 0, 0) in engine.active_coords
    print("   ✅ Initial matter added")
    
    # 2. Step
    print("   Running step()...")
    count = engine.step()
    
    print(f"   Matter count after step: {count}")
    # With MockModel returning x + 0.1, energy should increase and spread
    # The chunk logic should have activated neighbors
    
    assert count >= 1
    print("   ✅ Step executed")
    
    # 3. Viewport
    print("   Getting viewport tensor...")
    viewport = engine.get_viewport_tensor((0, 0, 0), 32, 0.0)
    print(f"   Viewport shape: {viewport.shape}")
    
    assert viewport.shape == (32, 32, d_state)
    print("   ✅ Viewport generated")
    
    print("✅ SparseHarmonicEngine Test Passed")

if __name__ == "__main__":
    test_harmonic_engine()
