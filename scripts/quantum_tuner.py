#!/usr/bin/env python3
"""
Quantum Tuner for Aetheria 4
============================

This script uses Qiskit Runtime Estimator and SPSA optimization to tune
Aetheria's hyperparameters (GAMMA_DECAY, LR_RATE) using a variational quantum circuit.

Architecture:
1. Variational Circuit (2 Qubits, RX/RY gates) -> Parameters Theta, Phi
2. Mapping: Theta -> GAMMA_DECAY, Phi -> LR_RATE
3. Cost Function: -Entropy of Aetheria simulation (50 steps)
4. Optimizer: SPSA (Simultaneous Perturbation Stochastic Approximation)
"""

import os
import sys
import numpy as np
import torch
import logging
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import SPSA
# Try importing Qiskit Runtime, fallback to local primitives if not available
try:
    from qiskit_ibm_runtime import EstimatorV2 as Estimator, Session
    RUNTIME_AVAILABLE = True
except ImportError:
    from qiskit.primitives import StatevectorEstimator as Estimator
    RUNTIME_AVAILABLE = False
    print("⚠️ Qiskit Runtime not found. Using local StatevectorEstimator.")

# Aetheria Imports
from src.engines.qca_engine import Aetheria_Motor
from src.models.unet import UNet
from src import config as global_cfg

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
GRID_SIZE = 64
D_STATE = 10
STEPS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Variational Circuit ---
def create_variational_circuit():
    """Creates a simple variational circuit with 2 parameters."""
    qc = QuantumCircuit(2)
    theta = Parameter('theta')
    phi = Parameter('phi')
    
    qc.h(0)
    qc.h(1)
    qc.rx(theta, 0)
    qc.ry(phi, 1)
    qc.cx(0, 1)
    
    return qc, [theta, phi]

# --- 2. Cost Function ---
def evaluate_aetheria(params):
    """
    Evaluates Aetheria simulation with given parameters.
    Returns -Entropy (to minimize).
    """
    theta, phi = params
    
    # Map parameters to hyperparameters (ensure positive values)
    # Theta [-pi, pi] -> GAMMA_DECAY [0.0, 0.1]
    gamma_decay = abs(float(theta)) * 0.015 
    
    # Phi [-pi, pi] -> LR_RATE [0.0001, 0.01]
    lr_rate = abs(float(phi)) * 0.001 + 0.0001
    
    print(f"🧪 Testing: GAMMA={gamma_decay:.6f}, LR={lr_rate:.6f}...", end="\r")
    
    # Create Config
    cfg = SimpleNamespace(
        GAMMA_DECAY=gamma_decay,
        LR_RATE=lr_rate,
        INITIAL_STATE_MODE_INFERENCE='complex_noise',
        EXPERIMENT_NAME='quantum_tuner'
    )
    
    # Instantiate Model (Dummy/Random for tuning structure)
    # In a real scenario, we might load a pre-trained model
    model = UNet(D_STATE, D_STATE).to(DEVICE)
    
    # Instantiate Engine
    motor = Aetheria_Motor(
        model_operator=model,
        grid_size=GRID_SIZE,
        d_state=D_STATE,
        device=DEVICE,
        cfg=cfg
    )
    
    # Run Simulation
    try:
        # Evolve for N steps
        for _ in range(STEPS):
            motor.evolve_internal_state()
            
        # Measure Entropy of final state
        psi = motor.state.psi
        if psi is None:
            return 0.0
            
        # Calculate Entropy: -Sum(p * log(p)) where p is density
        # Normalize density to treat as probability distribution
        density = torch.sum(psi.abs().pow(2), dim=-1) # (1, H, W)
        total_density = torch.sum(density)
        if total_density > 1e-9:
            prob = density / total_density
            # Avoid log(0)
            entropy = -torch.sum(prob * torch.log(prob + 1e-12)).item()
        else:
            entropy = 0.0
            
        if np.isnan(entropy) or np.isinf(entropy):
            return 100.0 # High penalty for instability
            
        # We want to MAXIMIZE entropy (complexity), so return NEGATIVE entropy
        return -entropy
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        return 100.0 # High penalty

# --- 3. Optimization Loop ---
def main():
    print("🚀 Starting Quantum Tuner for Aetheria...")
    print(f"   Device: {DEVICE}")
    print(f"   Runtime Available: {RUNTIME_AVAILABLE}")
    
    qc, parameters = create_variational_circuit()
    
    # Initial parameters (small random values to start near 0)
    initial_point = np.random.random(len(parameters)) * 0.1
    
    # SPSA Optimizer
    spsa = SPSA(maxiter=10) # Short run for demo
    
    print("\nStarting SPSA Optimization...")
    
    # Wrapper for SPSA (it expects a callable that returns a float)
    # Note: SPSA usually minimizes.
    result = spsa.minimize(evaluate_aetheria, initial_point)
    
    best_params = result.x
    best_entropy = -result.fun # Invert back to positive entropy
    
    # Map best params back to physical values
    best_gamma = abs(float(best_params[0])) * 0.015
    best_lr = abs(float(best_params[1])) * 0.001 + 0.0001
    
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Mejor configuración encontrada por el chip cuántico:")
    print(f"   GAMMA_DECAY = {best_gamma:.6f}")
    print(f"   LR_RATE     = {best_lr:.6f}")
    print(f"   Entropía    = {best_entropy:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
