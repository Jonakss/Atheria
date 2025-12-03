import sys
import os
import time
import torch

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.engines.native_engine_wrapper import NativeEngineWrapper

def test_native_quick():
    print("Quick Native Engine Test")
    print("=" * 50)
    
    grid_size = 16
    d_state = 64
    device = "cpu"
    
    print(f"Creating Native Engine (grid={grid_size}, d_state={d_state})...")
    engine = NativeEngineWrapper(grid_size=grid_size, d_state=d_state, device=device)
    
    model_path = "dummy_native_model_128ch.pt"
    print(f"Loading model: {model_path}...")
    if not engine.load_model(model_path):
        print(f"ERROR: Failed to load model")
        return False
    
    print("Running 5 steps...")
    for i in range(5):
        start = time.perf_counter()
        engine.evolve_internal_state(step=i)
        elapsed = time.perf_counter() - start
        print(f"  Step {i+1}/5 - {elapsed*1000:.2f}ms")
    
    print("✅ Native Engine completed 5 steps successfully!")
    return True

if __name__ == "__main__":
    success = test_native_quick()
    sys.exit(0 if success else 1)
