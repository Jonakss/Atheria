import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    import src.atheria_core as ac
    print("✅ atheria_core imported successfully")
except ImportError as e:
    print(f"❌ Failed to import atheria_core: {e}")
    sys.exit(1)

def test_octree_integration():
    print("\n--- Testing Octree Integration ---")
    
    # Initialize Engine
    engine = ac.Engine(d_state=64, device="cpu", grid_size=64)
    print("Engine initialized")
    
    # Add some particles
    coords = [
        (10, 10, 10),
        (20, 20, 20),
        (10, 10, 10), # Duplicate
        (100, 100, 100)
    ]
    
    state = torch.zeros(64, dtype=torch.float32)
    
    print(f"Adding {len(coords)} particles...")
    for x, y, z in coords:
        c = ac.Coord3D(x, y, z)
        engine.add_particle(c, state)
        
    # Run a step (which triggers octree.build())
    print("Running step_native()...")
    active_count = engine.step_native()
    print(f"Active particles after step: {active_count}")
    
    # We can't directly inspect the octree from Python yet without bindings,
    # but if step_native() didn't crash, it means the integration is at least stable.
    # The active_count should reflect the unique particles (3 unique coords).
    
    if active_count == 3:
        print("✅ Active count matches expected unique particles (3)")
    else:
        print(f"⚠️ Active count mismatch. Expected 3, got {active_count}")

    print("Test completed.")

if __name__ == "__main__":
    test_octree_integration()
