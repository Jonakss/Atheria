
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
    # Viewport A: Top-Left
    vp_a = (20, 20, 100, 100) # x1, y1, x2, y2

    # Viewport B: Center (Unobserved initially)
    # We won't observe this yet.

    # 4. Trigger Collapse (Observation)
    print(f"👁️  Observing Region A: {vp_a}")
    state_a = observer.get_viewport_state(vp_a)

    print(f"   Collapsed State Shape: {state_a.shape}")
    print(f"   Mag Mean: {state_a.abs().mean().item():.4f}")

    # 5. Visualize Results
    # We will plot:
    # 1. The Background Mean (Upscaled for comparison)
    # 2. The Collapsed Mask
    # 3. The Actual Collapsed State (Real part of Ch 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Background (Fog)
    bg_mu = observer.background_stats[..., :D_STATE]
    # Take channel 0
    bg_img = bg_mu[0, :, :, 0].cpu().numpy()
    axes[0].imshow(bg_img, cmap='viridis')
    axes[0].set_title("Background 'Fog' (Mu)")

    # Plot 2: Collapse Mask
    mask = observer.collapse_mask[0].cpu().numpy()
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Collapse Mask (Observed Regions)")

    # Plot 3: Full State (Real Reality)
    # We show the full buffer, but only observed parts are valid
    full_state_real = observer.collapsed_state.real[0, :, :, 0].cpu().numpy()
    # Mask out unobserved for clarity (or show zeros)
    # full_state_real[~mask] = 0 # Optional

    im3 = axes[2].imshow(full_state_real, cmap='plasma')
    axes[2].set_title("Collapsed Reality (High Res)")

    plt.tight_layout()
    output_path = 'experiment_06_observer_effect.png'
    plt.savefig(output_path)
    print(f"📸 Results saved to {output_path}")

    # Verify logic
    assert observer.collapse_mask[..., 20:100, 20:100].all(), "Region A should be fully collapsed"
    assert not observer.collapse_mask[..., 150:200, 150:200].any(), "Unobserved region should not be collapsed"

    print("✅ Experiment Success: Collapse logic verified.")

if __name__ == "__main__":
    run_experiment()
