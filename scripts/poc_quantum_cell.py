import torch
import pennylane as qml
import numpy as np
import sys
import os

# Add src to path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qca_engine_pennylane import QuantumCell, QuantumKernel

def test_quantum_cell_conservation():
    print("=== Test 1: Quantum Cell Probability Conservation ===")
    n_qubits = 4
    cell = QuantumCell(n_qubits=n_qubits, n_layers=2)

    # Random Input (normalized data for Angle Embedding, usually [0, pi])
    inputs = torch.rand(1, n_qubits) * np.pi
    inputs.requires_grad = True

    # Forward Pass
    probs = cell(inputs)

    print(f"Input: {inputs.detach().numpy()}")
    print(f"Output Probs: {probs.detach().numpy()}")
    print(f"Sum of Probs: {probs.sum().item():.6f}")

    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5), "Probabilities do not sum to 1!"
    print("SUCCESS: Unitarity Preserved (Sum = 1).")

    # Backward Pass (Check Gradients)
    loss = probs[0, 0] # Try to maximize probability of state |0000>
    loss.backward()

    print(f"Input Gradients: {inputs.grad}")
    assert inputs.grad is not None, "Gradients not flowing!"
    print("SUCCESS: Gradients Flowing.")
    print("-" * 30)

def test_quantum_kernel_convolution():
    print("=== Test 2: Quantum Kernel (Sliding Window) ===")
    # 3x3 kernel, 1 channel input
    kernel = QuantumKernel(n_qubits=9, n_layers=1)

    # Dummy Image (Batch=1, Channels=1, Height=6, Width=6)
    img = torch.rand(1, 1, 6, 6) * np.pi

    # Forward Pass
    print("Running Quantum Kernel on 6x6 grid... (This simulates 36 quantum circuits)")
    output = kernel(img)

    print(f"Input Shape: {img.shape}")
    print(f"Output Shape: {output.shape}")

    assert output.shape == (1, 1, 6, 6), f"Shape mismatch! Expected (1, 1, 6, 6), got {output.shape}"
    print("SUCCESS: Output shape is correct.")
    print("-" * 30)

if __name__ == "__main__":
    test_quantum_cell_conservation()
    test_quantum_kernel_convolution()
