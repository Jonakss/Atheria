import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import logging
from datetime import datetime
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.qca_engine import CartesianEngine, QuantumState
from src.physics.quantum_collapse import IonQCollapse
from src.physics.steering import QuantumSteering
from src.physics.quantum_kernel import QuantumMicroscope
from src.models.unet_unitary import UNetUnitary
from scripts.quantum_tuner import QuantumTuner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_lifecycle():
    print("🚀 Starting Quantum Lifecycle Experiment (Training -> Tuning -> Inference)...")
    
    # Config
    GRID_SIZE = 16
    D_STATE = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 2
    STEPS_PER_EPOCH = 10
    
    # ==============================================================================
    # PHASE 1: HYBRID TRAINING (Quantum-Aware)
    # ==============================================================================
    print("\n🏋️ PHASE 1: Hybrid Training (Learning to survive Quantum Collapse)...")
    
    # 1. Initialize Model
    model = UNetUnitary(d_state=D_STATE, hidden_channels=16).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 2. Initialize Engine & Collapser
    engine = CartesianEngine(model, GRID_SIZE, D_STATE, DEVICE)
    
    # Use Mock Collapser for Training Speed
    collapser = IonQCollapse(DEVICE)
    collapser.collapse = MagicMock(side_effect=collapser._mock_collapse) # Force mock
    engine.ionq_collapse = collapser
    
    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Init State
        psi = QuantumState(GRID_SIZE, D_STATE, DEVICE, initial_mode='complex_noise').psi
        psi.requires_grad_(True)
        
        loss_accum = 0
        
        # Evolve with Hybrid Steps
        for step in range(STEPS_PER_EPOCH):
            # Inject collapse every 3 steps
            psi = engine.evolve_hybrid_step(psi, step_num=step, injection_interval=3, noise_rate=0.1)
            
            # Loss: Maximize Complexity (Variance) + Minimize Energy Drift (Stability)
            # Simple proxy loss
            energy = psi.abs().pow(2).sum()
            target_energy = GRID_SIZE * GRID_SIZE * D_STATE * 0.5 # Arbitrary target
            loss_energy = (energy - target_energy).pow(2)
            
            loss_accum += loss_energy
            
        loss = loss_accum / STEPS_PER_EPOCH
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")
        
    # Save Model
    torch.save(model.state_dict(), "quantum_lifecycle_model.pth")
    print("   ✅ Model Trained and Saved.")
    
    # ==============================================================================
    # PHASE 2: QUANTUM TUNING (Finding Best Initial State)
    # ==============================================================================
    print("\n🎛️ PHASE 2: Quantum Tuning (Optimizing Initialization)...")
    
    model.eval()
    # Initialize Tuner with TRAINED model
    tuner = QuantumTuner(model=model, grid_size=GRID_SIZE, d_state=D_STATE, device=DEVICE, max_iter=2)
    tuner.tune()
    
    with open('best_quantum_params.json', 'r') as f:
        best_params = json.load(f)
    print(f"   ✅ Tuned State Found. Score: {best_params['best_score']:.4f}")
    
    # ==============================================================================
    # PHASE 3: INTERACTIVE INFERENCE (Steering + Microscope)
    # ==============================================================================
    print("\n🔮 PHASE 3: Interactive Inference (Steering & Microscope)...")
    
    # 1. Setup Inference Engine
    psi_init = QuantumState.create_variational_state(
        GRID_SIZE, D_STATE, DEVICE, best_params['best_params'], strength=0.8
    )
    engine = CartesianEngine(model, GRID_SIZE, D_STATE, DEVICE, precomputed_state=psi_init)
    
    # 2. Setup Tools
    steering = QuantumSteering(DEVICE)
    microscope = QuantumMicroscope(DEVICE)
    
    # 3. Inference Loop
    results_log = []
    
    for step in range(20):
        # Evolve
        engine.evolve_internal_state()
        
        # --- INTERACTION: STEERING at Step 5 ---
        if step == 5:
            print(f"   Step {step}: Injecting Quantum Vortex...")
            psi_before = engine.state.psi.clone()
            engine.state.psi = steering.inject(engine.state.psi, 'vortex', GRID_SIZE//2, GRID_SIZE//2)
            diff = (engine.state.psi - psi_before).abs().sum().item()
            print(f"      -> State Perturbation: {diff:.4f}")
            results_log.append({"step": step, "event": "steering", "delta": diff})
            
        # --- ANALYSIS: MICROSCOPE at Step 15 ---
        if step == 15:
            print(f"   Step {step}: Analyzing Center Patch...")
            # Extract center 4x4 patch
            center = GRID_SIZE // 2
            print(f"      -> PSI Shape: {engine.state.psi.shape}")
            # PSI is (Batch, Height, Width, Channels)
            patch = engine.state.psi[:, center-2:center+2, center-2:center+2, :]
            print(f"      -> Patch Shape: {patch.shape}")
            
            try:
                metrics = microscope.analyze_patch(patch)
                print(f"      -> Complexity: {metrics['complexity']:.4f}")
                print(f"      -> Activity: {metrics['activity']:.4f}")
                results_log.append({"step": step, "event": "microscope", "metrics": metrics})
            except Exception as e:
                print(f"      -> Microscope Error: {e}")
                
    # Save Results
    with open("docs/40_Experiments/lifecycle_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
        
    print("\n🎉 Quantum Lifecycle Experiment Complete!")
    print(f"   Results saved to docs/40_Experiments/lifecycle_results.json")

if __name__ == "__main__":
    run_lifecycle()
