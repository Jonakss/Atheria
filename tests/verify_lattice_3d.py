import sys
import os
import torch

# quick hack to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from engines.lattice_engine import LatticeEngine

def test_lattice_3d():
    print("Initializing LatticeEngine in 3D Mode...")
    # Initialize with depth=16, d_state=37
    engine = LatticeEngine(grid_size=32, d_state=37, depth=16, device="cpu")
    
    # Check State Shape
    assert hasattr(engine, 'ort_state'), "Engine should have 'ort_state'"
    print(f"ORT State Shape: {engine.ort_state.shape}")
    assert engine.ort_state.shape == (1, 37, 16, 32, 32)
    
    # Run Step
    print("Running step()...")
    engine.step()
    
    # Check output
    viz_data = engine.get_visualization_data("volumetric")
    print(f"Volumetric Viz Data Shape: {viz_data['shape']}")
    # Expect [Depth, Height, Width] (16, 32, 32) for scalar volumetric view
    assert viz_data['shape'] == [16, 32, 32]
    assert viz_data['channels'] == 1
    
    print("LatticeEngine 3D Verification Passed!")

if __name__ == "__main__":
    test_lattice_3d()
