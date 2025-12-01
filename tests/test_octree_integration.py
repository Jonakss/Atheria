import torch
import sys
import os
import pytest

# Add project root to path
# Add project root to path
sys.path.append(os.getcwd())
print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")
if os.path.exists("src"):
    print(f"src contents: {os.listdir('src')}")
else:
    print("src dir not found")

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

def test_octree_query_radius():
    print("\n--- Testing Octree Query Radius ---")
    
    # Initialize Engine
    engine = ac.Engine(d_state=64, device="cpu", grid_size=64)
    
    # Add particles
    # Center at (10, 10, 10)
    center = ac.Coord3D(10, 10, 10)
    engine.add_particle(center, torch.zeros(64))
    
    # Neighbors within radius 1
    n1 = ac.Coord3D(11, 10, 10)
    n2 = ac.Coord3D(9, 10, 10)
    engine.add_particle(n1, torch.zeros(64))
    engine.add_particle(n2, torch.zeros(64))
    
    # Far away particle
    far = ac.Coord3D(20, 20, 20)
    engine.add_particle(far, torch.zeros(64))
    
    # Force build octree (usually happens in step, but let's see if we can trigger it or if query handles it)
    # The current implementation builds octree in step_native. 
    # Let's run a dummy step to build it.
    engine.step_native()
    
    # Query radius 1 around center
    # Should return center, n1, n2 (3 particles)
    # Box query: [9, 11] in x, [9, 11] in y, [9, 11] in z
    results = engine.query_radius(center, 1)
    
    print(f"Query results count: {len(results)}")
    for p in results:
        print(f"  Found: ({p.x}, {p.y}, {p.z})")
        
    assert len(results) == 3, f"Expected 3 particles, got {len(results)}"
    
    # Verify coordinates
    coords = {(p.x, p.y, p.z) for p in results}
    assert (10, 10, 10) in coords
    assert (11, 10, 10) in coords
    assert (9, 10, 10) in coords
    assert (20, 20, 20) not in coords
    
    print("✅ query_radius test passed")

def test_step_native_execution():
    print("\n--- Testing Step Native Execution (Morton Order) ---")
    # This test mainly checks that we didn't break step_native with the sorting logic
    
    engine = ac.Engine(d_state=64, device="cpu", grid_size=64)
    
    # Add a block of particles
    for x in range(5):
        for y in range(5):
            engine.add_particle(ac.Coord3D(x, y, 0), torch.randn(64))
            
    initial_count = engine.get_matter_count()
    print(f"Initial particles: {initial_count}")
    
    # Run step
    engine.step_native()
    
    final_count = engine.get_matter_count()
    print(f"Final particles: {final_count}")
    
    # Without a model, particles should persist (if logic in step_native handles no-model case correctly)
    # The C++ code says: if (!model_loaded_) return matter_map_.size();
    # So it should be equal.
    assert final_count == initial_count
    
    print("✅ step_native execution passed")

if __name__ == "__main__":
    test_octree_query_radius()
    test_step_native_execution()
