import torch
import unittest
from src.engines.holographic_mixin import HolographicMixin

class TestHolographicMixin(unittest.TestCase):
    def test_generate_bulk_state(self):
        mixin = HolographicMixin()
        
        # Test Case 1: Standard 2D field
        # Shape keys: [B, C, H, W] -> [1, 1, 64, 64]
        H, W = 64, 64
        base_field = torch.rand(1, 1, H, W)
        depth = 8
        
        bulk = mixin._generate_bulk_state(base_field, depth)
        
        # Expected shape: [1, D, H, W]
        self.assertEqual(bulk.shape, (1, depth, H, W))
        self.assertAlmostEqual(bulk.min().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(bulk.max().item(), 1.0, delta=0.01)
        
        print(f"✅ Test 1 Passed: {bulk.shape}")

    def test_engine_integration(self):
        # Mock engine utilizing the mixin
        class MockEngine(HolographicMixin):
            def __init__(self):
                self.bulk_depth = 5
            
            def get_dense_state(self):
                # Returns [1, H, W, C]
                return torch.randn(1, 32, 32, 1)

            def get_visualization_data(self, viz_type):
                if viz_type == 'holographic_bulk':
                    return self.get_bulk_visualization_data(
                        viz_type, 
                        lambda: self.get_dense_state().permute(0, 3, 1, 2), # Correct to [1, C, H, W] for mixin
                        self.bulk_depth
                    )
        
        engine = MockEngine()
        result = engine.get_visualization_data('holographic_bulk')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'holographic_bulk')
        self.assertTrue(result['is_volumetric'])
        self.assertEqual(result['shape'], [5, 32, 32])
        self.assertEqual(len(result['data']), 5 * 32 * 32)
        
        print("✅ Test 2 Passed: Engine Integration")

if __name__ == '__main__':
    unittest.main()
