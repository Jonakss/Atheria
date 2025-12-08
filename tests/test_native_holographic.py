import torch
import unittest
import numpy as np
try:
    from src.engines.native_engine_wrapper import NativeEngineWrapper, NATIVE_AVAILABLE
except ImportError:
    NATIVE_AVAILABLE = False

class TestNativeHolographic(unittest.TestCase):
    def setUp(self):
        # Initialize Wrapper (CPU for testing)
        self.grid_size = 64
        self.d_state = 1
        try:
            # Force CPU to trigger the wrapper's fallback logic for linking errors
            self.engine = NativeEngineWrapper(self.grid_size, self.d_state, device='cpu')
        except Exception as e:
            self.skipTest(f"Failed to init native engine (even on CPU fallback): {e}")

    def test_native_bulk_generation(self):
        # 1. Add some particles
        self.engine.add_initial_particles(num_particles=20)
        
        # 2. Request Holographic Bulk Viz
        result = self.engine.get_visualization_data('holographic_bulk')
        
        # 3. Verify Structure
        self.assertIsNotNone(result, "Result should not be None")
        self.assertIsInstance(result, dict)
        self.assertIn('data', result)
        self.assertIn('shape', result)
        self.assertTrue(result['is_volumetric'])
        self.assertEqual(result['engine'], 'native')
        
        # 4. Verify Shape [D, H, W]
        depth, h, w = result['shape']
        self.assertEqual(h, self.grid_size)
        self.assertEqual(w, self.grid_size)
        self.assertGreater(depth, 1) # Should be 5 or 8
        
        # 5. Verify Data consistency
        flat_data = result['data']
        self.assertEqual(len(flat_data), depth * h * w)
        
        print(f"✅ Native Bulk Test Passed. Shape: {result['shape']}")

    def tearDown(self):
        if hasattr(self, 'engine'):
            self.engine.cleanup()

if __name__ == '__main__':
    unittest.main()
