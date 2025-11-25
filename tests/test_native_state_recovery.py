#!/usr/bin/env python3
"""
Test script to diagnose add_particle() / get_state_at() issue in the native engine.
"""
import sys
import os
import torch
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_native_state_recovery():
    """Diagnose native engine particle storage and recovery."""
    print("="*80)
    print("TEST: Native State Recovery (Diagnosis)")
    print("="*80)

    try:
        import atheria_core
        print("✅ atheria_core module imported successfully")
    except ImportError as e:
        print(f"❌ Error importing atheria_core: {e}")
        return False

    # Configuration
    d_state = 1
    grid_size = 16
    device = "cpu"  # Keep it simple for diagnosis

    print(f"📊 Configuration: grid_size={grid_size}, d_state={d_state}, device={device}")

    try:
        engine = atheria_core.Engine(d_state, device, grid_size)
        print(f"✅ Native engine created successfully")
    except Exception as e:
        print(f"❌ Error creating native engine: {e}")
        return False

    # 1. Add 10 particles manually
    print("\nStep 1: Adding 10 particles manually...")
    particles_added = 0
    test_coords = []

    for i in range(10):
        x = (i * 3) % grid_size
        y = (i * 5) % grid_size
        z = 0

        # Create a simple state (1.0 + 0j)
        state = torch.ones(d_state, dtype=torch.complex64, device=device)
        coord = atheria_core.Coord3D(x, y, z)

        try:
            engine.add_particle(coord, state)
            test_coords.append(coord)
            particles_added += 1
            print(f"  Added particle at ({x}, {y})")
        except Exception as e:
            print(f"  ❌ Error adding particle at ({x}, {y}): {e}")

    print(f"✅ Added {particles_added} particles.")

    # 2. Verify get_matter_count()
    print("\nStep 2: Verifying matter count...")
    try:
        matter_count = engine.get_matter_count()
        print(f"📊 get_matter_count() returns: {matter_count}")

        if matter_count == particles_added:
            print("✅ Matter count matches added particles.")
        else:
            print(f"❌ Matter count MISMATCH. Expected {particles_added}, got {matter_count}.")
            if matter_count == 0:
                print("⚠️ CRITICAL: Engine reports 0 particles. add_particle() failed to store.")
    except Exception as e:
        print(f"❌ Error getting matter count: {e}")

    # 3. Retrieve each particle with get_state_at()
    print("\nStep 3: Retrieving particles with get_state_at()...")
    recovered_count = 0

    for i, coord in enumerate(test_coords):
        try:
            state = engine.get_state_at(coord)
            if state is not None:
                max_val = state.abs().max().item()
                if max_val > 0.9: # We added ones, so should be around 1.0
                    print(f"  ✅ Particle {i} at ({coord.x}, {coord.y}): Recovered (val={max_val:.2f})")
                    recovered_count += 1
                else:
                     print(f"  ⚠️ Particle {i} at ({coord.x}, {coord.y}): Empty state (val={max_val:.6f})")
            else:
                print(f"  ❌ Particle {i} at ({coord.x}, {coord.y}): get_state_at returned None")
        except Exception as e:
             print(f"  ❌ Error retrieving particle {i}: {e}")

    print(f"\n📊 Recovery Summary: {recovered_count}/{particles_added} particles recovered.")

    if recovered_count == particles_added:
        print("\n✅ DIAGNOSIS: Native engine is working correctly.")
        return True
    else:
        print("\n❌ DIAGNOSIS: Native engine has storage/retrieval issues.")
        return False

if __name__ == "__main__":
    test_native_state_recovery()
