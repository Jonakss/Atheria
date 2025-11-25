import torch
import sys
import os
import time
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
        # Simple diffusion + persistence + growth
        # x is [1, C, H, W]
        return x * 0.9 + 0.05

def test_infinite_universe():
    print("🌌 Testing Infinite Universe Simulation (SparseHarmonicEngine)...")
    
    device = "cpu"
    d_state = 8
    model = MockModel(d_state).to(device)
    
    engine = SparseHarmonicEngine(model, d_state, device, grid_size=256)
    
    # 1. Inject Genesis Seed
    print("   🌱 Injecting Genesis Seed...")
    initial_state = torch.randn(d_state, device=device)
    engine.add_matter(0, 0, 0, initial_state)
    
    print(f"   Initial matter count: {len(engine.matter)}")
    
    # 2. Run Simulation Loop
    print("   🚀 Running simulation loop (10 steps)...")
    start_time = time.time()
    
    for i in range(10):
        count = engine.step()
        print(f"   Step {i+1}: {count} particles")
        
    end_time = time.time()
    print(f"   ✅ Simulation finished in {end_time - start_time:.2f}s")
    
    # 3. Verify Expansion
    assert len(engine.matter) > 1, "Universe did not expand!"
    print("   ✅ Universe expansion confirmed")
    
    # 4. Verify Viewport Generation
    print("   🔭 Generating viewport...")
    viewport = engine.get_dense_state()
    print(f"   Viewport shape: {viewport.shape}")
    assert viewport.shape == (256, 256, d_state)
    print("   ✅ Viewport generated successfully")

if __name__ == "__main__":
    test_infinite_universe()
