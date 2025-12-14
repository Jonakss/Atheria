import torch
import sys
import os

# quick hack to find modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from qca.observer_effect import ObserverKernel

def test_observer():
    kernel = ObserverKernel()
    B, C, D, H, W = 1, 37, 32, 32, 32
    state = torch.ones(B, C, D, H, W)
    
    # Generate Mask
    mask = kernel.get_observer_mask((B, C, D, H, W))
    print(f"Mask Shape: {mask.shape}")
    assert mask.shape == (B, 1, D, H, W)
    
    # Collapse
    collapsed = kernel.collapse(state, mask)
    print(f"Collapsed Shape: {collapsed.shape}")
    assert collapsed.shape == (B, C, D, H, W)
    
    # Verify masking (center should be 1, corners 0)
    # Center
    assert collapsed[0, 0, D//2, H//2, W//2] > 0.9, "Center should be observed"
    # Corner
    assert collapsed[0, 0, 0, 0, 0] == 0, "Corner should be fog (0 in MVP)"
    
    print("Observer Kernel Test Passed!")

if __name__ == "__main__":
    test_observer()
