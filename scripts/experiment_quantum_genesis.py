#!/usr/bin/env python3
"""
Experiment 008: Quantum Genesis Simulation
==========================================

This script runs a full simulation initialized with "Quantum Genesis" (IonQ).
It tracks the evolution of the universe starting from a true quantum seed.

Metrics tracked:
- Entropy over time
- Total Energy over time
- Particle Count (if applicable)

Usage:
    export IONQ_API_KEY="your_key"
    python3 scripts/experiment_quantum_genesis.py
"""

import sys
import os
import time
import torch
import numpy as np
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.qca_engine import CartesianEngine
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_entropy(psi: torch.Tensor) -> float:
    """Calculates Shannon entropy of the state."""
    prob = psi.abs().pow(2)
    prob = prob / (prob.sum() + 1e-9)
    p_flat = prob.flatten()
    p_flat = p_flat[p_flat > 1e-9]
    entropy = -torch.sum(p_flat * torch.log(p_flat)).item()
    return entropy

def run_experiment(steps=100, grid_size=128, d_state=4):
    print(f"\n🚀 Starting Experiment 008: Quantum Genesis Simulation")
    print(f"   Grid: {grid_size}x{grid_size}, Channels: {d_state}, Steps: {steps}")
    print(f"   Backend: {config.IONQ_BACKEND_NAME}")

    # 1. Initialize Engine with IonQ
    print("\n⚛️ Initializing Universe (Quantum Genesis)...")
    start_init = time.time()
    
    try:
        # We use a dummy model for CartesianEngine as we just want to see evolution dynamics
        # In a real scenario, this would be a trained model.
        # For this experiment, we'll use a random unitary model to simulate physics
        model = torch.nn.Conv2d(d_state*2, d_state*2, kernel_size=3, padding=1, bias=False)
        # Initialize with orthogonal weights to be somewhat unitary/stable
        torch.nn.init.orthogonal_(model.weight)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        engine = CartesianEngine(model, grid_size, d_state, device, initial_mode='ionq')
        
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    init_time = time.time() - start_init
    print(f"✅ Initialization Complete in {init_time:.2f}s")
    
    # Initial Metrics
    metrics = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "grid_size": grid_size,
            "steps": steps,
            "init_time": init_time,
            "backend": config.IONQ_BACKEND_NAME
        },
        "history": []
    }
    
    # 2. Simulation Loop
    print("\n🔄 Starting Simulation Loop...")
    start_sim = time.time()
    
    for step in range(steps):
        # Evolve
        engine.evolve_internal_state(step=step)
        
        # Measure
        psi = engine.state.psi
        entropy = calculate_entropy(psi)
        energy = psi.abs().pow(2).sum().item()
        
        # Record
        step_data = {
            "step": step + 1,
            "entropy": entropy,
            "energy": energy
        }
        metrics["history"].append(step_data)
        
        if step % 10 == 0:
            print(f"   Step {step+1}/{steps}: Entropy={entropy:.4f}, Energy={energy:.4f}")

    total_sim_time = time.time() - start_sim
    print(f"\n✅ Simulation Complete in {total_sim_time:.2f}s")
    print(f"   Avg Step Time: {total_sim_time/steps*1000:.2f}ms")
    
    # 3. Save Results
    output_file = "experiment_quantum_genesis_results.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"💾 Results saved to {output_file}")
    
    # 4. Analysis Summary
    initial_entropy = metrics["history"][0]["entropy"]
    final_entropy = metrics["history"][-1]["entropy"]
    print("\n🧐 Quick Analysis:")
    print(f"   Entropy Change: {initial_entropy:.4f} -> {final_entropy:.4f}")
    if final_entropy > initial_entropy:
        print("   -> Complexity Increased (Emergence?)")
    else:
        print("   -> Complexity Decreased (Dissipation?)")

if __name__ == "__main__":
    run_experiment()
