
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from engines.lattice_engine import LatticeEngine
from config import Config

def test_bulk_gen():
    print("Initializing LatticeEngine...")
    cfg = Config()
    # Use smaller grid for speed
    cfg.GRID_SIZE = 32
    engine = LatticeEngine(cfg)
    
    print("Generating Bulk Data...")
    try:
        viz_data = engine.get_visualization_data("holographic_bulk")
        data = viz_data['data']
        meta = viz_data['metadata']
        
        print(f"Success! Data Shape: {data.shape}")
        print(f"Metadata Shape: {meta['shape']}")
        
        if len(data.shape) == 3 and data.shape[0] == 8:
            print("VERIFICATION PASSED: Shape is [D=8, H, W]")
        else:
            print("VERIFICATION FAILED: Incorrect shape")
            
    except Exception as e:
        print(f"VERIFICATION FAILED: Exception raised: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bulk_gen()
