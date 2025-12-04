import torch
import sys
import os
from qiskit import QuantumCircuit

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_quantum_native_training import QuantumNativeConv2d

def main():
    checkpoint_path = "output/checkpoints/quantum_native_model.pt"
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found")
        return

    cp = torch.load(checkpoint_path, map_location='cpu')
    state_dict = cp['model_state_dict']
    
    # Reconstruct model
    # We need to know grid_size and n_layers. 
    # We can infer n_qubits from theta shape: [n_layers, n_qubits]
    theta = state_dict['theta']
    n_layers, n_qubits = theta.shape
    grid_size = int(2**(n_qubits/2)) # Assuming square grid
    
    print(f"Reconstructing model: Grid={grid_size}x{grid_size}, Layers={n_layers}")
    
    model = QuantumNativeConv2d(grid_size=grid_size, n_layers=n_layers, device='cpu')
    model.load_state_dict(state_dict)
    
    qc = model.export_circuit()
    
    print("\n--- QASM START ---")
    try:
        # Try QASM 2 (Legacy but standard)
        print(qc.qasm())
    except Exception as e:
        print(f"QASM 2 failed: {e}")
        try:
            # Try QASM 3
            from qiskit import qasm3
            print(qasm3.dumps(qc))
        except Exception as e2:
            print(f"QASM 3 failed: {e2}")
            print("Printing circuit draw instead:")
            print(qc.draw(fold=80))
    print("--- QASM END ---")

if __name__ == "__main__":
    main()
