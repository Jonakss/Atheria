import torch
import logging
import sys
import os
import unittest
from unittest.mock import MagicMock

# Force local import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.qca_engine import CartesianEngine, QuantumState
from src.engines.harmonic_engine import SparseHarmonicEngine as HarmonicEngine
# NativeEngineWrapper might fail if no C++ module, so we mock or skip if import fails
try:
    from src.engines.native_engine_wrapper import NativeEngineWrapper
except ImportError:
    NativeEngineWrapper = None

logging.basicConfig(level=logging.INFO)

class TestQuantumTools(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.grid_size = 64
        self.d_state = 2

    def test_cartesian_engine_tools(self):
        print("\n🧪 Testing CartesianEngine Tools...")
        # Mock model operator
        model = MagicMock()
        engine = CartesianEngine(model, self.grid_size, self.d_state, self.device)
        
        # Initialize state
        engine.state.psi = torch.randn(1, self.grid_size, self.grid_size, self.d_state, dtype=torch.complex64)
        initial_psi = engine.state.psi.clone()
        
        # Apply Collapse
        success = engine.apply_tool('collapse', {'intensity': 0.5})
        self.assertTrue(success)
        self.assertFalse(torch.allclose(engine.state.psi, initial_psi), "State should change after collapse")
        print("✅ CartesianEngine Collapse passed")

    def test_harmonic_engine_tools(self):
        print("\n🧪 Testing HarmonicEngine Tools...")
        model = MagicMock()
        engine = HarmonicEngine(model, self.d_state, self.device, grid_size=self.grid_size)
        
        # Apply Collapse (Injection)
        success = engine.apply_tool('collapse', {'intensity': 0.5, 'x': 32, 'y': 32})
        self.assertTrue(success)
        self.assertTrue(len(engine.matter) > 0, "Matter should be injected")
        print("✅ HarmonicEngine Collapse passed")

if __name__ == '__main__':
    unittest.main()
