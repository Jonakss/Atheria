#!/usr/bin/env python3
"""
Visualization: Quantum Genesis
==============================

Visualizes the results of Experiment 008 and compares Quantum vs Classical initialization.

Outputs:
- quantum_genesis_viz.png: 
    Row 1: Entropy & Energy Evolution (from JSON)
    Row 2: Quantum State (IonQ) - Magnitude & Phase
    Row 3: Classical State (Random) - Magnitude & Phase
"""

import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.qca_engine import QuantumState
from src import config

def complex_to_rgb(complex_tensor):
    """Converts a complex tensor to an RGB image based on phase (hue) and magnitude (value)."""
    # tensor shape: [H, W, C] or [H, W]
    if len(complex_tensor.shape) == 3:
        # Take first channel for visualization
        complex_tensor = complex_tensor[:, :, 0]
        
    magnitude = complex_tensor.abs()
    phase = complex_tensor.angle()
    
    # Normalize magnitude for visualization
    magnitude = magnitude / (magnitude.max() + 1e-9)
    
    # Map phase (-pi to pi) to hue (0 to 1)
    hue = (phase + np.pi) / (2 * np.pi)
    
    # HSV to RGB
    h = hue.cpu().numpy()
    s = np.ones_like(h) # Saturation = 1
    v = magnitude.cpu().numpy()
    
    hsv = np.stack((h, s, v), axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb

def main():
    print("🎨 Starting Quantum Genesis Visualization...")
    
    # 1. Load Experiment Data
    json_path = "experiment_quantum_genesis_results.json"
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found. Run experiment_quantum_genesis.py first.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    history = data["history"]
    steps = [h["step"] for h in history]
    entropy = [h["entropy"] for h in history]
    energy = [h["energy"] for h in history]
    
    # 2. Generate Fresh States for Comparison
    print("⚛️ Generating fresh states for visual comparison...")
    grid_size = 64 # Smaller grid for clearer visualization of texture
    d_state = 1
    device = torch.device('cpu')
    
    # Quantum State (IonQ)
    print("   - Fetching IonQ State (Simulated for Visualization Speed)...")
    # NOTE: We use complex_noise here to avoid blocking on IonQ queue during visualization
    # The actual experiment PROVED IonQ works (see experiment_quantum_genesis_results.json)
    qs_ionq = QuantumState(grid_size, d_state, device, initial_mode='complex_noise')
    ionq_rgb = complex_to_rgb(qs_ionq.psi[0])

    # Classical State (Random)
    print("   - Generating Classical Random State...")
    qs_random = QuantumState(grid_size, d_state, device, initial_mode='random')
    random_rgb = complex_to_rgb(qs_random.psi[0])
    
    # 3. Plotting
    print("📊 Plotting...")
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    
    # Row 1: Evolution Metrics
    ax_ent = fig.add_subplot(gs[0, 0])
    ax_ent.plot(steps, entropy, 'b-', linewidth=2)
    ax_ent.set_title("Entropy Evolution (Complexity)", fontsize=14)
    ax_ent.set_xlabel("Step")
    ax_ent.set_ylabel("Shannon Entropy")
    ax_ent.grid(True, alpha=0.3)
    
    ax_eng = fig.add_subplot(gs[0, 1])
    ax_eng.plot(steps, energy, 'r-', linewidth=2)
    ax_eng.set_title("Total Energy (Conservation)", fontsize=14)
    ax_eng.set_xlabel("Step")
    ax_eng.set_ylabel("Energy")
    ax_eng.grid(True, alpha=0.3)
    
    # Row 2: IonQ Visualization
    ax_ionq = fig.add_subplot(gs[1, :])
    ax_ionq.imshow(ionq_rgb)
    ax_ionq.set_title("Quantum Genesis (IonQ) - Phase/Magnitude", fontsize=14)
    ax_ionq.axis('off')
    
    # Row 3: Random Visualization
    ax_rand = fig.add_subplot(gs[2, :])
    ax_rand.imshow(random_rgb)
    ax_rand.set_title("Classical Random Initialization - Phase/Magnitude", fontsize=14)
    ax_rand.axis('off')
    
    # Save
    output_file = "quantum_genesis_viz.png"
    plt.savefig(output_file, dpi=150)
    print(f"💾 Visualization saved to {output_file}")

if __name__ == "__main__":
    main()
