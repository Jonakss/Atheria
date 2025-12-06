# src/models/quantum_kernel.py
"""
Quantum Kernel - Modelo basado en circuitos cuánticos (PennyLane).

Este módulo define capas de red neural que internamente usan
circuitos cuánticos variaciones (VQC) de PennyLane.

NOTA: Este es un STUB - la implementación completa requiere:
1. Instalar PennyLane: pip install pennylane pennylane-qiskit
2. Implementar el circuito variacional
3. Conectar con backends (simulado o real)

ARQUITECTURA:
- Los engines (Cartesian, Polar, etc.) usan un MODELO como "Ley M"
- Este modelo puede ser: UNet, UNetUnitary, o QuantumKernel
- El modelo define cómo evoluciona el estado: delta_psi = model(psi)

USO FUTURO:
    from src.models.quantum_kernel import QuantumKernel
    
    model = QuantumKernel(n_qubits=4, n_layers=2)
    engine = CartesianEngine(model, grid_size, d_state, device)
"""
import torch
import torch.nn as nn
import logging


class QuantumKernel(nn.Module):
    """
    Quantum Kernel usando PennyLane VQC.
    
    STUB: Actualmente solo pasa el input sin modificar.
    La implementación completa usará pennylane.qnode para
    definir un circuito cuántico diferenciable.
    
    Args:
        n_qubits: Número de qubits en el circuito
        n_layers: Número de capas variaciones (profundidad)
        d_state: Dimensión del estado (para compatibilidad con engines)
    """
    
    _compiles = False  # No usar torch.compile con circuitos cuánticos
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, d_state: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.d_state = d_state
        
        # TODO: Implementar circuito PennyLane
        # Ejemplo de estructura:
        # import pennylane as qml
        # dev = qml.device("default.qubit", wires=n_qubits)
        # 
        # @qml.qnode(dev, interface="torch", diff_method="backprop")
        # def circuit(inputs, weights):
        #     qml.AngleEmbedding(inputs, wires=range(n_qubits))
        #     for layer in range(n_layers):
        #         qml.StronglyEntanglingLayers(weights[layer], wires=range(n_qubits))
        #     return qml.expval(qml.PauliZ(0))
        #
        # self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        
        logging.warning("QuantumKernel: STUB - PennyLane no implementado. Retornando entrada sin modificar.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del kernel cuántico.
        
        Args:
            x: Input tensor [B, C, H, W] (formato de engines)
            
        Returns:
            Output tensor [B, C, H, W] (delta_psi para el engine)
            
        TODO: Implementar:
        1. Reducir dimensionalidad (x es HxW píxeles)
        2. Pasar por circuito cuántico
        3. Expandir a la dimensionalidad original
        """
        # STUB: Solo retorna la entrada
        return x
    
    def get_params_count(self) -> int:
        """Retorna el número de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters())
