import unittest
import numpy as np
from scripts.poc_qdesic import classical_velocity_sq, qdesic_velocity_sq, gr_velocity_sq

class TestQdesicPoC(unittest.TestCase):

    def test_classical_velocity_newtonian(self):
        """
        Test that classical velocity matches Newtonian expectation (Lambda=0).
        v^2 = GM/r
        """
        G, M = 1.0, 1.0
        Lambda = 0.0
        r = 10.0
        expected_v2 = G * M / r
        result_v2 = classical_velocity_sq(r, G, M, Lambda)
        self.assertAlmostEqual(result_v2, expected_v2)

    def test_qdesic_reduces_to_gr(self):
        """
        Test that qdesic_velocity_sq reduces to GR (gr_velocity_sq) when all epsilon params are 0.
        Note: We avoid r=2GM (singularities) by choosing r=10.0.
        """
        G, M = 1.0, 1.0
        Lambda = -0.001
        r = np.array([10.0, 20.0, 50.0])

        eps_params = {'e00':0, 'e10':0, 'e02':0, 'e12':0, 'e22':0}

        v2_gr = gr_velocity_sq(r, G, M, Lambda)
        v2_qdesic = qdesic_velocity_sq(r, G, M, Lambda, eps_params)

        np.testing.assert_allclose(v2_qdesic, v2_gr, rtol=1e-10)

    def test_qdesic_modification(self):
        """
        Test that non-zero epsilon parameters produce a deviation from GR.
        """
        G, M = 1.0, 1.0
        Lambda = -0.001
        r = 10.0

        # All zero case (GR)
        eps_zero = {'e00':0, 'e10':0, 'e02':0, 'e12':0, 'e22':0}
        v2_zero = qdesic_velocity_sq(r, G, M, Lambda, eps_zero)

        # Modified case
        eps_mod = {'e00':0.1, 'e10':0, 'e02':0, 'e12':0, 'e22':0}
        v2_mod = qdesic_velocity_sq(r, G, M, Lambda, eps_mod)

        self.assertNotEqual(v2_zero, v2_mod, "Non-zero epsilon should modify velocity")

    def test_division_by_zero_handling(self):
        """
        Ensure potential singularities are handled.
        """
        G, M = 1.0, 1.0
        Lambda = 0
        r = np.array([2.0]) # r=2GM is singularity for GR formula

        # Should return NaN or large value, but not crash
        result = qdesic_velocity_sq(r, G, M, Lambda)
        self.assertTrue(isinstance(result, (np.ndarray, float)))
        if isinstance(result, np.ndarray):
             self.assertTrue(np.isnan(result[0]) or np.isinf(result[0]) or abs(result[0]) > 1e6)

if __name__ == '__main__':
    unittest.main()
