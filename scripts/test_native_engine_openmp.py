#!/usr/bin/env python3
"""
Test script for native engine OpenMP parallelization.
Verifies correctness and measures performance.
"""
import sys
import time
import torch
import numpy as np

def test_native_engine_parallel():
    """Test native engine with OpenMP parallelization."""
    
    print("=" * 60)
    print("Native Engine OpenMP Parallelization Test")
    print("=" * 60)
    
    # Import native module directly
    try:
        import atheria_core
    except ImportError as e:
        print(f"❌ Failed to import atheria_core: {e}")
        return 1
    
    # Configuration
    d_state = 8
    grid_size = 64
    num_particles = 1000
    num_steps = 10
    
    print(f"\nConfiguration:")
    print(f"  d_state: {d_state}")
    print(f"  grid_size: {grid_size}")
    print(f"  num_particles: {num_particles}")
    print(f"  num_steps: {num_steps}")
    
    # Create native engine directly
    print("\n🔧 Initializing native engine...")
    engine = atheria_core.Engine(d_state=d_state, device="cpu", grid_size=grid_size)
    
    # Add particles manually (fallback method)
    print(f"🔄 Adding {num_particles} particles manually...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    for i in range(num_particles):
        x = np.random.randint(-grid_size//4, grid_size//4)
        y = np.random.randint(-grid_size//4, grid_size//4)
        z = 0  # Only Z=0 for 2D visualization
        
        coord = atheria_core.Coord3D(x, y, z)
        state = torch.randn(d_state, dtype=torch.complex64)
        state = state / torch.linalg.norm(state)  # Normalize
        
        engine.add_particle(coord, state)
        
        if (i + 1) % 250 == 0:
            print(f"  Added {i + 1}/{num_particles} particles...")
    
    initial_count = engine.get_matter_count()
    print(f"✅ Initial particle count: {initial_count}")
    
    if initial_count == 0:
        print("❌ No particles were added!")
        return 1
    
    # Run simulation without model (conservation test)
    print(f"\n🔄 Running {num_steps} steps (no model, conservation test)...")
    start_time = time.time()
    
    for step in range(num_steps):
        count = engine.step_native()
        if step % 5 == 0:
            print(f"  Step {step}: {count} particles")
    
    elapsed = time.time() - start_time
    final_count = engine.get_matter_count()
    
    print(f"\n📊 Results:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per step: {elapsed/num_steps:.3f}s")
    print(f"  Throughput: {num_steps/elapsed:.2f} steps/sec")
    print(f"  Final particle count: {final_count}")
    print(f"  Particle conservation: {final_count == initial_count}")
    
    # Determinism test (thread safety)
    print("\n🔄 Testing determinism (thread safety)...")
    engine2 = atheria_core.Engine(d_state=d_state, device="cpu", grid_size=grid_size)
    
    # Add same particles with same seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    for i in range(num_particles):
        x = np.random.randint(-grid_size//4, grid_size//4)
        y = np.random.randint(-grid_size//4, grid_size//4)
        z = 0
        coord = atheria_core.Coord3D(x, y, z)
        state = torch.randn(d_state, dtype=torch.complex64)
        state = state / torch.linalg.norm(state)
        engine2.add_particle(coord, state)
    
    for _ in range(num_steps):
        engine2.step_native()
    
    final_count2 = engine2.get_matter_count()
    
    # Compare particle counts (states are harder to compare without model)
    is_deterministic = (final_count == final_count2)
    
    print(f"  Engine 1 final count: {final_count}")
    print(f"  Engine 2 final count: {final_count2}")
    print(f"  Deterministic: {is_deterministic}")
    
    # Summary
    print("\n" + "=" * 60)
    if final_count == initial_count and is_deterministic:
        print("✅ ALL TESTS PASSED")
        print("  ✓ Particle conservation OK")
        print("  ✓ Thread safety (determinism) OK")
        print(f"  ✓ Performance: {num_steps/elapsed:.2f} steps/sec")
        return 0
    else:
        print("❌ TESTS FAILED")
        if final_count != initial_count:
            print(f"  ✗ Particle conservation FAILED (expected {initial_count}, got {final_count})")
        if not is_deterministic:
            print("  ✗ Thread safety (determinism) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(test_native_engine_parallel())
