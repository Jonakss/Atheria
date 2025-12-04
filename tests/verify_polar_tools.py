
import torch
import torch.nn as nn
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)

from src.engines.qca_engine_polar import PolarEngine

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)
        
    def forward(self, x):
        return self.conv(x)

def verify_polar_engine():
    print("🧪 Verifying PolarEngine Tools & Interface...")
    
    device = 'cpu' # Use CPU for testing
    grid_size = 64
    d_state = 1
    
    model = DummyModel()
    engine = PolarEngine(model, grid_size, d_state, device)
    
    # 1. Verify get_dense_state
    print("\n1. Testing get_dense_state()...")
    dense_state = engine.get_dense_state()
    print(f"   Shape: {dense_state.shape}")
    print(f"   Type: {dense_state.dtype}")
    
    if dense_state.shape != (1, grid_size, grid_size, d_state):
        print("❌ Shape mismatch! Expected (1, H, W, C)")
        return False
        
    if not torch.is_complex(dense_state):
        print("❌ Type mismatch! Expected complex tensor")
        return False
        
    print("✅ get_dense_state() passed.")
    
    # 2. Verify apply_tool (Collapse)
    print("\n2. Testing apply_tool('collapse')...")
    initial_mag = engine.state.psi.magnitude.clone()
    
    # Apply collapse
    success = engine.apply_tool('collapse', {'intensity': 0.8, 'x': 32, 'y': 32})
    
    if not success:
        print("❌ apply_tool('collapse') returned False")
        return False
        
    final_mag = engine.state.psi.magnitude
    diff = (final_mag - initial_mag).abs().sum().item()
    print(f"   Magnitude difference: {diff}")
    
    if diff == 0:
        print("❌ State did not change after collapse!")
        return False
        
    print("✅ apply_tool('collapse') passed.")
    
    # 3. Verify apply_tool (Vortex)
    print("\n3. Testing apply_tool('vortex')...")
    initial_phase = engine.state.psi.phase.clone()
    
    # Apply vortex
    success = engine.apply_tool('vortex', {'radius': 10, 'strength': 1.0, 'x': 32, 'y': 32})
    
    if not success:
        print("❌ apply_tool('vortex') returned False")
        return False
        
    final_phase = engine.state.psi.phase
    diff_phase = (final_phase - initial_phase).abs().sum().item()
    print(f"   Phase difference: {diff_phase}")
    
    if diff_phase == 0:
        print("❌ Phase did not change after vortex!")
        return False
        
    print("✅ apply_tool('vortex') passed.")
    
    print("\n🎉 All PolarEngine tests passed!")
    return True

if __name__ == "__main__":
    try:
        if verify_polar_engine():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Exception during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
