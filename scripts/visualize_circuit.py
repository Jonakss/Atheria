import torch
import matplotlib.pyplot as plt
import os
import sys
from qiskit import QuantumCircuit

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_advanced_ansatz import StronglyEntanglingConv2d

def main():
    model_path = "output/models/quantum_fastforward_final.pt"
    output_dir = "docs/40_Experiments/images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp009_circuit.png")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Load Model
    cp = torch.load(model_path, map_location='cpu')
    config = cp['config']
    grid_size = config['grid_size']
    n_layers = config['n_layers']
    
    model = StronglyEntanglingConv2d(grid_size=grid_size, n_layers=n_layers, device='cpu')
    model.load_state_dict(cp['model_state_dict'])
    
    # Export Circuit
    qc = model.export_circuit()
    
    # Draw
    print("Drawing circuit...")
    # Use matplotlib output
    qc.draw(output='mpl', filename=output_path)
    print(f"Circuit diagram saved to: {output_path}")

if __name__ == "__main__":
    main()
