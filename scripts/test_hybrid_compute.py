import sys
import os
import torch
import logging
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.qca_engine import CartesianEngine
from src.physics.quantum_collapse import IonQCollapse

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_hybrid_compute():
    print("⚡ Testing Hybrid Compute (IonQCollapse)...")
    
    device = torch.device('cpu')
    
    # 1. Test IonQCollapse Module directly
    print("   1. Testing IonQCollapse module...")
    collapser = IonQCollapse(device)
    
    # Mock backend to avoid API calls and check logic
    with patch('src.engines.compute_backend.IonQBackend') as MockBackend:
        mock_instance = MockBackend.return_value
        # Mock execute return
        mock_instance.execute.return_value = {'11111111111': 1}
        
        # Inject mock backend
        collapser.backend = mock_instance
        
        # Create dummy state
        state = torch.zeros(1, 32, 32, 4, dtype=torch.complex64)
        
        # Run collapse
        new_state = collapser.collapse(state, region_center=(16, 16), intensity=1.0)
        
        # Check if state changed in the region
        # We expect the center 11 pixels to have changed magnitude
        center_slice = new_state[0, 16, 16:16+11, :]
        mag = center_slice.abs().sum().item()
        
        print(f"      - Region Magnitude after collapse: {mag:.4f}")
        
        if mag > 0:
            print("      ✅ Collapse modified the state.")
        else:
            print("      ❌ Collapse did not modify the state.")
            
    # 2. Test Integration in Engine
    print("   2. Testing Engine Integration...")
    model = MagicMock()
    # Fix: Return tuple (output, h_state, c_state) because engine expects it if has_memory is True
    # Or better, ensure has_memory is False for this test
    # But MagicMock might trigger has_memory=True check if it has 'convlstm' attribute (it doesn't by default)
    # The error was: ValueError: not enough values to unpack (expected 3, got 1)
    # This means engine thought it had memory. Let's check why.
    # CartesianEngine checks: hasattr(model_operator, 'convlstm') or 'ConvLSTM' in model_operator.__class__.__name__
    # MagicMock doesn't have convlstm.
    # Ah, wait. The error is in _evolve_logic:
    # if self.has_memory: ...
    # Let's force has_memory=False by ensuring the mock doesn't look like ConvLSTM
    
    # Actually, let's just make the mock return what's expected if it WERE treated as memory, just in case
    # But wait, if has_memory is False, it shouldn't unpack 3 values.
    # Line 452: delta_psi_unitario_complex, h_next, c_next = self.operator(x_cat_total, h_t, c_t)
    # This line is inside `if self.has_memory:`.
    # So self.has_memory MUST be True.
    # Why?
    # self.has_memory = hasattr(model_operator, 'convlstm') or 'ConvLSTM' in model_operator.__class__.__name__
    # MagicMock name is ... MagicMock.
    
    # Let's debug by printing has_memory in the script or just fixing the return.
    # Simplest fix: Make the mock return a tuple, covering both cases if possible, 
    # OR explicitly set engine.has_memory = False after init.
    
    # Fix: Model output must be 2 * d_state channels (Real + Imag parts)
    # d_state = 4 -> Output channels = 8
    model.return_value = torch.zeros(1, 8, 32, 32) 
    engine = CartesianEngine(model, 32, 4, device)
    engine.has_memory = False # Force disable memory for this test
    
    # Force inject our collapser (with mock)
    engine.ionq_collapse = collapser
    
    # Run hybrid step
    psi_in = torch.zeros(1, 32, 32, 4, dtype=torch.complex64)
    psi_out = engine.evolve_hybrid_step(psi_in, step_num=10, injection_interval=10)
    
    # Check if collapse was called
    # Since we mocked the backend inside collapser, and passed collapser to engine,
    # if engine called collapser.collapse, the state should be modified.
    
    diff = (psi_out - psi_in).abs().sum().item()
    print(f"      - Step Difference: {diff:.4f}")
    
    if diff > 0:
        print("      ✅ Engine successfully triggered hybrid collapse.")
    else:
        print("      ❌ Engine did not trigger collapse (or interval mismatch).")

if __name__ == "__main__":
    test_hybrid_compute()
