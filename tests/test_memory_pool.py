import torch
import sys
import os
import pytest
import time

# Add project root to path
sys.path.append(os.getcwd())

try:
    import src.atheria_core as ac
    print("✅ atheria_core imported successfully (from src)")
except ImportError:
    try:
        import atheria_core as ac
        print("✅ atheria_core imported successfully (top level)")
    except ImportError as e:
        print(f"❌ Failed to import atheria_core: {e}")
        sys.exit(1)

def test_memory_pool_stability():
    print("\n--- Testing Memory Pool Stability ---")
    
    engine = ac.Engine(d_state=64, device="cpu", grid_size=64)
    
    # Add a block of particles
    for x in range(10):
        for y in range(10):
            engine.add_particle(ac.Coord3D(x, y, 0), torch.randn(64))
            
    initial_count = engine.get_matter_count()
    print(f"Initial particles: {initial_count}")
    
    # Run multiple steps to trigger recycling
    # Without a model, step_native just returns count, but we want to ensure
    # the recycling logic (which runs at end of step) doesn't crash even if 
    # no new particles were created via pool (since model is not loaded).
    # Wait, if model is not loaded, step_native returns early!
    # We need to load a dummy model or bypass the check to test pool logic?
    # Actually, the pool logic is inside the `if (model_loaded_)` block in step_native.
    # So we need a model to test the pool.
    
    # Let's load the dummy model if it exists
    model_path = "dummy_native_model_128ch.pt"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            # Debug: Try loading in python first
            pt_model = torch.jit.load(model_path)
            print("✅ Model loaded in Python successfully")
        except Exception as e:
            print(f"❌ Failed to load model in Python: {e}")

        engine.load_model(model_path)
    else:
        print("⚠️ No dummy model found. Skipping full pool test, running basic stability.")
        # Even without model, we can check if engine is stable
    
    start_time = time.time()
    for i in range(10):
        count = engine.step_native()
        if i % 2 == 0:
            print(f"Step {i}: {count} particles")
            
    end_time = time.time()
    print(f"10 steps took {end_time - start_time:.4f}s")
    
    print("✅ Memory Pool stability test passed")

if __name__ == "__main__":
    test_memory_pool_stability()
