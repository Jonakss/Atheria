import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import os

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
        # Check if we want to use IonQ
        if "ionq" in dev_name:
            api_key = os.getenv("IONQ_API_KEY")
            if not api_key:
                print("⚠️ IONQ_API_KEY not found. Falling back to default.qubit")
                self.dev = qml.device("default.qubit", wires=n_qubits)
            else:
                try:
                    # 'ionq.simulator' or 'ionq.qpu'
                    # If dev_name is just 'ionq', default to simulator
                    device_target = "ionq.simulator" if dev_name == "ionq" else dev_name
                    print(f"🔌 Initializing IonQ Device: {device_target}")
                    self.dev = qml.device(device_target, wires=n_qubits, shots=1024, api_key=api_key)
                except Exception as e:
                    print(f"❌ Error initializing IonQ device: {e}. Falling back to default.qubit")
                    self.dev = qml.device("default.qubit", wires=n_qubits)
        else:
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
        # However, TorchLayer expects (Batch, Input_Size)
        # So we can't easily vectorize the whole image in one go purely inside TorchLayer
        # unless we treat each pixel's neighborhood as a sample in the batch.

        # 1. Unfold
        # x_unfold: (Batch, C*K*K, H_out*W_out)
        # We use padding=1 to keep dimensions
        inp_unfold = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, padding=1)

        # 2. Reshape for QNode
        # (Batch, 9, H*W) -> (Batch * H * W, 9)
        # We process all pixels as a large batch
        inputs = inp_unfold.transpose(1, 2).reshape(-1, self.kernel_size**2)

        # 3. Run Quantum Layer
        # (Batch * H * W, n_actions)
        q_out = self.q_layer(inputs)

        # 4. Reshape back
        # (Batch, H*W, n_actions) -> (Batch, n_actions, H, W)
        q_out = q_out.view(b, h, w, -1).permute(0, 3, 1, 2)

        return q_out

class QuantumCell(nn.Module):
    """
    Simulates a single Quantum Cell evolving unitarily.
    Useful for the POC script.
    """
    def __init__(self, n_qubits=4, n_layers=2, dev_name="default.qubit"):
        super().__init__()

        if "ionq" in dev_name:
            api_key = os.getenv("IONQ_API_KEY")
            if not api_key:
                 self.dev = qml.device("default.qubit", wires=n_qubits)
            else:
                 try:
                    device_target = "ionq.simulator" if dev_name == "ionq" else dev_name
                    self.dev = qml.device(device_target, wires=n_qubits, shots=1024, api_key=api_key)
                 except:
                    self.dev = qml.device("default.qubit", wires=n_qubits)
        else:
            self.dev = qml.device(dev_name, wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits)) # Return full probability distribution

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x)
