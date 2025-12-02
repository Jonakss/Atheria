import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.batch_inference_engine import BatchInferenceEngine
from src.engines.qca_engine import CartesianEngine, QuantumState
from src import config as cfg

def complex_to_rgb(psi_complex):
    """Convierte estado complejo a RGB (Fase -> Hue, Magnitud -> Value)"""
    amplitude = np.abs(psi_complex)
    phase = np.angle(psi_complex)
    
    # Normalizar amplitud para visualización
    amplitude = amplitude / (amplitude.max() + 1e-9)
    
    # HSV a RGB manual simplificado
    # H = phase, S = 1, V = amplitude
    # Usamos matplotlib hsv colormap
    import matplotlib.colors as mcolors
    
    # Normalizar fase a [0, 1]
    phase_norm = (phase + np.pi) / (2 * np.pi)
    
    # Crear imagen HSV
    hsv = np.zeros((psi_complex.shape[0], psi_complex.shape[1], 3))
    hsv[..., 0] = phase_norm
    hsv[..., 1] = 1.0 # Saturation
    hsv[..., 2] = amplitude # Value
    
    return mcolors.hsv_to_rgb(hsv)

def visualize_multiverse():
    print("🌌 Generating Quantum Multiverse Visualization...")
    
    # Mock IonQ Backend for visualization speed/cost
    with patch('src.engines.compute_backend.IonQBackend') as MockBackend:
        mock_instance = MockBackend.return_value
        # Return 3 distinct seeds to show divergence
        mock_instance.execute.return_value = {
            '00000000000': 1, # Seed A
            '11111111111': 1, # Seed B
            '01010101010': 1  # Seed C
        }
        
        # Setup Engine
        grid_size = 64
        d_state = 4
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mock Model (Identity + diffusion for visual effect)
        # We need a model that actually evolves the state to show divergence
        # Let's use a simple function instead of a full NN for this viz script
        class MockModel(torch.nn.Module):
            def forward(self, x):
                # Simple diffusion/reaction simulation
                # x: [batch, 2*d_state, H, W]
                # Return same shape as input (2*d_state) to match expected output for complex chunking
                return x * 0.1 # Small change
                
        model = MockModel()
        
        engine = BatchInferenceEngine(model, grid_size, d_state, device)
        
        # Initialize 3 universes
        num_universes = 3
        engine.initialize_from_ionq_multiverse(num_universes, strength=0.5)
        
        # Evolve for some steps
        steps_to_viz = [0, 10, 50]
        history = {0: [], 10: [], 50: []}
        
        # Save step 0
        for i in range(num_universes):
            history[0].append(engine.states[i].psi.cpu().numpy()[0, :, :, 0]) # Channel 0
            
        # Evolve to 10
        engine.evolve_batch(steps=10)
        for i in range(num_universes):
            history[10].append(engine.states[i].psi.cpu().numpy()[0, :, :, 0])
            
        # Evolve to 50
        engine.evolve_batch(steps=40)
        for i in range(num_universes):
            history[50].append(engine.states[i].psi.cpu().numpy()[0, :, :, 0])
            
        # Plotting
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        titles = ["Universe A (Seed 0...0)", "Universe B (Seed 1...1)", "Universe C (Seed 0...1)"]
        
        for row, step in enumerate(steps_to_viz):
            for col in range(num_universes):
                ax = axes[row, col]
                
                # Get state
                state_complex = history[step][col]
                rgb = complex_to_rgb(state_complex)
                
                ax.imshow(rgb)
                if row == 0:
                    ax.set_title(titles[col], fontsize=10, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f"Step {step}", fontsize=10, fontweight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle("Quantum Multiverse Divergence\n(Simulated Evolution from IonQ Seeds)", fontsize=16)
        
        output_path = "docs/assets/quantum_multiverse_viz.png"
        os.makedirs("docs/assets", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_multiverse()
