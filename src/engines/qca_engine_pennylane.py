import torch.nn as nn

class QuantumKernel(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
    def forward(self, x):
        return x
