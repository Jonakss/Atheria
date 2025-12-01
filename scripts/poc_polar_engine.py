import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qca_engine_polar import PolarEngine, QuantumStatePolar
from src.models.rotational_unet import RotationalUNet

def test_polar_engine():
    print("=== Testing Polar Engine ===")

    # 1. Setup
    model = RotationalUNet()
    engine = PolarEngine(model, grid_size=32)

    # 2. Init State (B, 1, 32, 32)
    B, H, W = 1, 32, 32
    magnitude = torch.rand(B, 1, H, W)
    phase = torch.rand(B, 1, H, W) * 2 * np.pi - np.pi # [-pi, pi]

    state = QuantumStatePolar(magnitude, phase)

    print(f"Initial Phase Range: [{phase.min().item():.3f}, {phase.max().item():.3f}]")

    # 3. Evolution Loop
    for step in range(5):
        state = engine(state)

        mag_min, mag_max = state.magnitude.min().item(), state.magnitude.max().item()
        phase_min, phase_max = state.phase.min().item(), state.phase.max().item()

        print(f"Step {step+1}:")
        print(f"  Mag Range: [{mag_min:.3f}, {mag_max:.3f}]")
        print(f"  Phase Range: [{phase_min:.3f}, {phase_max:.3f}]")

        # Verify Phase is within [-pi, pi]
        assert phase_min >= -np.pi - 1e-5, "Phase underflow!"
        assert phase_max <= np.pi + 1e-5, "Phase overflow!"

    print("SUCCESS: Engine runs and keeps phase bounded.")

if __name__ == "__main__":
    test_polar_engine()
