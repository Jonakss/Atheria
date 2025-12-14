
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, hsv_to_rgb

# Ensure src is in path
sys.path.append(os.getcwd())

from src.qca.observer_effect import ObserverEffect

def run_experiment():
    print("🔬 Experiment 06: Observer Effect (Holographic Collapse)")
    print("=======================================================")

    # Configuration
    GRID_SIZE = 256
    D_STATE = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    # 1. Initialize the Observer Engine
    observer = ObserverEffect(GRID_SIZE, D_STATE, DEVICE, downscale_factor=8)
    print("✅ Observer Engine Initialized.")
    print(f"   Background Size: {observer.bg_size}x{observer.bg_size}")

    # 2. Simulate Background Evolution (The Fog)
    print("🌫️  Simulating Background Evolution (100 steps)...")
    for _ in range(100):
        observer.update_background()

    # 3. Define Viewports (The Observation)
    # Use aligned coordinates (multiples of 8) to test clean unobserving
    vp_a = (16, 16, 112, 112) # x1, y1, x2, y2

    # 4. Trigger Collapse (Observation)
    print(f"👁️  Observing Region A: {vp_a}")
    state_a = observer.get_viewport_state(vp_a)

    # 5. Verify Unobserve Logic
    print(f"🌫️  Unobserving Region A...")
    observer.unobserve(vp_a)
    assert not observer.collapse_mask[..., 16:112, 16:112].any(), "Region A should be unobserved (fog)"

    print(f"👁️  Re-Observing Region A (Should sample from updated fog)...")
    state_a_new = observer.get_viewport_state(vp_a)

    # 6. Visualize Results (Orbital Style: Density vs Phase)

    # Get the state for visualization (using the viewport state)
    # We take channel 0 for simplicity
    psi = state_a_new[0, ..., 0].cpu() # [H_vp, W_vp] complex

    # Density (Magnitude)
    density = psi.abs()

    # Phase (Angle) mapped to HSV
    phase = psi.angle()

    # Create Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Background (Fog) - Showing Mean
    bg_mu = observer.background_stats[..., :D_STATE]
    bg_img = bg_mu[0, :, :, 0].cpu().numpy()
    axes[0].imshow(bg_img, cmap='viridis')
    axes[0].set_title("Background 'Fog' (Mean)")

    # Plot 2: Density (Orbital Probability)
    im2 = axes[1].imshow(density.numpy(), cmap='inferno') # Inferno is good for "brightness/density"
    axes[1].set_title("Density (Probability Cloud)")
    plt.colorbar(im2, ax=axes[1])

    # Plot 3: Phase (Resonance/Interference)
    # Map phase to Hue
    # H = (phase + pi) / 2pi, S=1, V=1 (or V=density)
    phase_norm = (phase.numpy() + np.pi) / (2 * np.pi)
    hsv = np.zeros((phase.shape[0], phase.shape[1], 3))
    hsv[..., 0] = phase_norm # Hue
    hsv[..., 1] = 1.0        # Saturation
    hsv[..., 2] = 1.0        # Value (Brightness) - Optional: mask by density

    # Optional: Mask by density so low-prob areas are dark
    # hsv[..., 2] = density.numpy() / density.max()

    rgb = hsv_to_rgb(hsv)

    axes[2].imshow(rgb)
    axes[2].set_title("Phase (Resonance Bands)")

    plt.tight_layout()
    output_path = 'experiment_06_observer_effect.png'
    plt.savefig(output_path)
    print(f"📸 Results saved to {output_path}")

    print("✅ Experiment Success: Collapse and Unobserve logic verified.")

if __name__ == "__main__":
    run_experiment()
