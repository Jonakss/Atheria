import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QuantumKernel(nn.Module):
    """
    A Quantum Convolutional Layer (Quantum Kernel).
    It acts as a sliding window of 3x3 cells.

    Architecture:
    1. Unfold image into 3x3 patches.
    2. Encode patch data into Qubits (Angle Embedding).
    3. Apply Variational Quantum Circuit (VQC).
    4. Measure Center Qubit (or all) to get new state.
    """
    def __init__(self, n_qubits=9, n_layers=2, n_actions=1, dev_name="lightning.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.kernel_size = 3

        # Define the device
        # We use 'lightning.qubit' for fast C++ simulation if available
        try:
            self.dev = qml.device(dev_name, wires=n_qubits)
        except:
            print(f"Device {dev_name} not found, falling back to default.qubit")
            self.dev = qml.device("default.qubit", wires=n_qubits)

        # Define the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Encode Classical Data (3x3 patch = 9 values)
            # We assume inputs are normalized [-pi, pi] or [0, pi]
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

            # Variational Layers (The "Quantum Law")
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Measure
            # We measure the expectation value of Z on the center qubit (wire 4 for 3x3)
            # Or we can measure all and sum/concat.
            # For this POC, let's return the expectation of the center qubit.
            return [qml.expval(qml.PauliZ(i)) for i in range(n_actions)]

        # Weight Shapes for StronglyEntanglingLayers
        # shape: (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # Create TorchLayer
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        """
        x: (Batch, Channels, Height, Width)
        We handle 1 channel for now. If multiple, we might need multiple kernels or parallel qubits.
        """
        b, c, h, w = x.shape

        # We assume c=1 for the 1-to-1 mapping of pixel to qubit.
        # If c > 1, we should probably reduce first or use more complex embedding.
        if c != 1:
            # For POC simplicity, take the mean across channels or raise warning
            # x = x.mean(dim=1, keepdim=True)
            pass

        # Unfold to get 3x3 patches
        # Output: (Batch, Channels*kernel*kernel, L) where L = H*W (if padding is correct)
        # We need padding=1 to keep same size
        unfold = nn.Unfold(kernel_size=self.kernel_size, padding=1)
        patches = unfold(x) # (B, 9, H*W)

        # Prepare for Quantum Layer
        # Reshape to (Batch * H * W, 9)
        patches_flat = patches.transpose(1, 2).reshape(-1, self.kernel_size**2 * c)

        # Run Quantum Circuit
        # Output: (Batch * H * W, n_actions)
        q_out = self.q_layer(patches_flat)

        # Reshape back to image
        # (Batch, n_actions, H, W)
        q_out = q_out.reshape(b, -1, h, w)

        return q_out

class QuantumCell(nn.Module):
    """
    Simulates a single Quantum Cell evolving unitarily.
    Useful for the POC script.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits)) # Return full probability distribution

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x)
