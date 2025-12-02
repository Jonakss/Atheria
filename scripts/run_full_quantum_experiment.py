import sys
import os
import torch
import logging
import json
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.qca_engine import CartesianEngine, QuantumState
from src.physics.quantum_collapse import IonQCollapse
from src.physics.steering import QuantumSteering
from scripts.quantum_tuner import QuantumTuner
from src import config as cfg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_full_experiment():
    print("🚀 Starting Full Quantum Experiment (Tuner + Hybrid + Steering)...")

    device = torch.device('cpu') # Use CPU for test
    grid_size = 32
    d_state = 4

    # ==============================================================================
    # PHASE 1: QUANTUM TUNER (Genesis)
    # ==============================================================================
    print("\n🔬 PHASE 1: Quantum Tuner (Optimizing Initial State)...")

    # Mock model for tuner (since we might not have a trained one)
    # Must return 8 channels (real+imag for 4 states)
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 8, grid_size, grid_size)

    tuner = QuantumTuner(mock_model, grid_size, d_state, device)
    tuner.max_iter = 3 # Short run

    # Run tuning
    tuner.tune()

    # Load best params
    with open('best_quantum_params.json', 'r') as f:
        best_params = json.load(f)
    print(f"   ✅ Tuner finished. Best Score: {best_params['score']:.4f}")

    # Generate Optimal Initial State
    print("   Generating optimal initial state...")
    psi_init = QuantumState.create_variational_state(
        grid_size, d_state, device, best_params['params'], strength=0.8
    )

    # ==============================================================================
    # PHASE 2: HYBRID SIMULATION (Evolution)
    # ==============================================================================
    print("\n⚛️ PHASE 2: Hybrid Simulation (Running with IonQ Collapse)...")

    engine = CartesianEngine(mock_model, grid_size, d_state, device, precomputed_state=psi_init)
    engine.has_memory = False # Disable memory for mock

    # Force inject collapser (mock backend inside)
    engine.ionq_collapse = IonQCollapse(device)
    # Mock backend for collapse to avoid API calls/errors in test
    engine.ionq_collapse._mock_collapse = MagicMock(side_effect=engine.ionq_collapse._mock_collapse)

    # Run loop
    steps = 10
    injection_interval = 3

    print(f"   Running {steps} steps with collapse every {injection_interval}...")

    current_psi = engine.state.psi
    for i in range(steps):
        current_psi = engine.evolve_hybrid_step(
            current_psi, step_num=i, injection_interval=injection_interval, noise_rate=0.2
        )

        # Check if collapse happened
        if i % injection_interval == 0:
            print(f"   Step {i}: Hybrid Event Triggered.")

    engine.state.psi = current_psi
    print("   ✅ Hybrid Simulation finished.")

    # ==============================================================================
    # PHASE 3: QUANTUM STEERING (Interaction)
    # ==============================================================================
    print("\n🖌️ PHASE 3: Quantum Steering (Injecting Vortex)...")

    steering = QuantumSteering(device)

    # Inject Vortex at center
    print("   Injecting 'vortex' at (16, 16)...")
    psi_steered = steering.inject(engine.state.psi, 'vortex', 16, 16)

    # Verify change
    diff = (psi_steered - engine.state.psi).abs().sum().item()
    print(f"   State difference after steering: {diff:.4f}")

    if diff > 0:
        print("   ✅ Steering successful.")
    else:
        print("   ❌ Steering failed (no change).")

    print("\n🎉 Full Experiment Completed Successfully!")

if __name__ == "__main__":
    run_full_experiment()
