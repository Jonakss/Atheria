import torch
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.harmonic_engine import HarmonicEngine
from src.engines.qca_engine import CartesianEngine
from src.engines.lattice_engine import LatticeEngine
# NativeEngineWrapper might fail if not compiled, so we wrap it
try:
    from src.engines.native_engine_wrapper import NativeEngineWrapper
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)

def test_engine_tools(engine_name, engine):
    print(f"\nTesting {engine_name}...")
    
    # Test Collapse
    print(f"  - Testing 'collapse'...")
    success = engine.apply_tool('collapse', {'x': 64, 'y': 64, 'intensity': 0.8})
    if success:
        print(f"    ✅ Collapse applied successfully.")
    else:
        print(f"    ❌ Collapse failed.")

    # Test Vortex
    print(f"  - Testing 'vortex'...")
    success = engine.apply_tool('vortex', {'x': 64, 'y': 64, 'radius': 10, 'strength': 1.0})
    if success:
        print(f"    ✅ Vortex applied successfully.")
    else:
        print(f"    ❌ Vortex failed.")
        
    # Test Wave
    print(f"  - Testing 'wave'...")
    success = engine.apply_tool('wave', {'k_x': 2.0, 'k_y': 2.0})
    if success:
        print(f"    ✅ Wave applied successfully.")
    else:
        print(f"    ❌ Wave failed.")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on {device}")
    
    # Mock Model for Cartesian/Harmonic
    model = torch.nn.Conv2d(16, 16, 3, padding=1).to(device)
    
    # 1. HarmonicEngine
    harmonic = HarmonicEngine(model, d_state=16, device=device, grid_size=128)
    harmonic.initialize_matter()
    test_engine_tools("HarmonicEngine", harmonic)
    
    # 2. CartesianEngine
    cartesian = CartesianEngine(model, grid_size=128, d_state=16, device=device)
    test_engine_tools("CartesianEngine", cartesian)
    
    # 3. LatticeEngine
    lattice = LatticeEngine(grid_size=128, d_state=16, device=device)
    test_engine_tools("LatticeEngine", lattice)
    
    # 4. NativeEngineWrapper
    if NATIVE_AVAILABLE:
        try:
            native = NativeEngineWrapper(grid_size=128, d_state=16, device='cpu') # Force CPU to avoid CUDA issues if any
            test_engine_tools("NativeEngineWrapper", native)
        except Exception as e:
            print(f"Skipping NativeEngineWrapper: {e}")
    else:
        print("Skipping NativeEngineWrapper (not available)")

if __name__ == "__main__":
    main()
