import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    checkpoint_path = "output/checkpoints/quantum_native_model.pt"
    output_dir = "docs/40_Experiments/images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp008_results.png")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location='cpu')
    
    losses = cp['losses']
    target_sample = cp['target_output_sample'].detach().numpy().flatten()
    pqc_sample = cp['pqc_output_sample'].detach().numpy().flatten()
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Curve
    axs[0].plot(losses, label='Training Loss')
    axs[0].set_title('Training Convergence (MSE)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # 2. Phase Distribution (Target vs PQC)
    # Extract phases
    target_phases = np.angle(target_sample)
    pqc_phases = np.angle(pqc_sample)
    
    # Sort for better visualization (or just histogram)
    axs[1].hist(target_phases, bins=50, alpha=0.5, label='Target Phase', color='blue')
    axs[1].hist(pqc_phases, bins=50, alpha=0.5, label='PQC Phase', color='orange')
    axs[1].set_title('Phase Distribution Histogram')
    axs[1].set_xlabel('Phase (radians)')
    axs[1].set_ylabel('Count')
    axs[1].legend()
    
    # 3. Phase Correlation (Scatter)
    # If perfect, should be a diagonal line
    # Downsample for scatter if too large
    indices = np.random.choice(len(target_phases), size=min(1000, len(target_phases)), replace=False)
    axs[2].scatter(target_phases[indices], pqc_phases[indices], alpha=0.5, s=10)
    axs[2].plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', alpha=0.5) # Diagonal
    axs[2].set_title('Phase Correlation (Target vs PQC)')
    axs[2].set_xlabel('Target Phase')
    axs[2].set_ylabel('PQC Phase')
    axs[2].set_xlim(-np.pi, np.pi)
    axs[2].set_ylim(-np.pi, np.pi)
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
