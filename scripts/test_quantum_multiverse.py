import sys
import os
import torch
import logging
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.batch_inference_engine import BatchInferenceEngine
from src.engines.qca_engine import CartesianEngine
from src import config as cfg

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_multiverse_initialization():
    print("🌌 Testing Quantum Multiverse Initialization...")
    
    # Mock IonQ Backend to avoid using credits
    with patch('src.engines.compute_backend.IonQBackend') as MockBackend:
        # Setup Mock
        mock_instance = MockBackend.return_value
        # Mock execute to return a distribution of bitstrings
        # Let's say we ask for 4 universes
        # We return 4 shots: two '00...0' and two '11...1' to see divergence
        mock_instance.execute.return_value = {
            '00000000000': 2,
            '11111111111': 2
        }
        
        # Setup Engine
        grid_size = 32
        d_state = 4
        device = torch.device('cpu')
        model = MagicMock() # Mock model
        
        engine = BatchInferenceEngine(model, grid_size, d_state, device)
        
        # Run Initialization
        num_universes = 4
        engine.initialize_from_ionq_multiverse(num_universes)
        
        # Verify
        print(f"   - Requested {num_universes} universes.")
        print(f"   - Created {len(engine.states)} states.")
        
        if len(engine.states) != num_universes:
            print("❌ Failed: Incorrect number of states.")
            return

        # Check divergence
        # Universe 0 and 1 should be identical (seed '00...0')
        # Universe 2 and 3 should be identical (seed '11...1')
        # Universe 0 and 2 should be different
        
        u0 = engine.states[0].psi
        u1 = engine.states[1].psi
        u2 = engine.states[2].psi
        
        diff_0_1 = torch.sum(torch.abs(u0 - u1)).item()
        diff_0_2 = torch.sum(torch.abs(u0 - u2)).item()
        
        print(f"   - Diff U0 vs U1 (Same Seed): {diff_0_1:.6f}")
        print(f"   - Diff U0 vs U2 (Diff Seed): {diff_0_2:.6f}")
        
        if diff_0_1 < 1e-5 and diff_0_2 > 1.0:
            print("✅ Success: Universes correctly seeded from quantum distribution.")
        else:
            print("❌ Failed: Divergence check failed.")

if __name__ == "__main__":
    test_multiverse_initialization()
