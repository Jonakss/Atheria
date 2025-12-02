#!/usr/bin/env python3
"""
Benchmark: Quantum Genesis vs Classical Initialization
======================================================

Comparativa de rendimiento y entropía entre inicialización clásica (pseudo-random)
y Quantum Genesis (IonQ).

Métricas:
1. Tiempo de inicialización (Latencia).
2. Entropía de Shannon del estado generado (Complejidad).
3. Distribución de energía.
"""

import sys
import os
import time
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.qca_engine import QuantumState
from src.engines.harmonic_engine import SparseHarmonicEngine
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_entropy(psi: torch.Tensor) -> float:
    """Calcula la entropía de Shannon del estado."""
    # Probabilidad = |psi|^2 normalizada
    prob = psi.abs().pow(2)
    prob = prob / (prob.sum() + 1e-9)
    
    # H = -sum(p * log(p))
    # Aplanar para cálculo
    p_flat = prob.flatten()
    p_flat = p_flat[p_flat > 1e-9] # Evitar log(0)
    
    entropy = -torch.sum(p_flat * torch.log(p_flat)).item()
    return entropy

def benchmark_engine(engine_name: str, mode: str, grid_size=64, d_state=4):
    """Ejecuta benchmark para un motor y modo específico."""
    device = torch.device('cpu') # Usar CPU para consistencia en medición de tiempo de red
    
    print(f"\n--- Benchmarking {engine_name} [{mode}] ---")
    
    start_time = time.time()
    
    state_tensor = None
    
    try:
        if engine_name == "Cartesian/Native":
            # QuantumState se usa tanto en Cartesian como Native
            qs = QuantumState(grid_size, d_state, device, initial_mode=mode)
            state_tensor = qs.psi
            
        elif engine_name == "Harmonic":
            # HarmonicEngine usa su propio método pero internamente llama a QuantumState
            # Necesitamos un modelo dummy para instanciarlo
            dummy_model = torch.nn.Linear(d_state, d_state).to(device)
            engine = SparseHarmonicEngine(dummy_model, d_state, device, grid_size)
            engine.initialize_matter(mode=mode)
            
            # Reconstruir tensor denso para medir entropía
            state_tensor = engine.get_viewport_tensor((0,0,0), grid_size, 0.0)
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

    elapsed = time.time() - start_time
    print(f"⏱️ Time: {elapsed:.4f}s")
    
    if state_tensor is not None:
        entropy = calculate_entropy(state_tensor)
        energy = state_tensor.abs().pow(2).sum().item()
        print(f"📊 Entropy: {entropy:.4f}")
        print(f"⚡ Total Energy: {energy:.4f}")
        
        return {
            "time": elapsed,
            "entropy": entropy,
            "energy": energy
        }
    return None

def main():
    print("🚀 Starting Quantum Genesis Benchmark...")
    print(f"IonQ Backend: {config.IONQ_BACKEND_NAME}")
    
    results = {}
    
    modes = ['complex_noise', 'ionq']
    engines = ["Cartesian/Native", "Harmonic"]
    
    for eng in engines:
        for mode in modes:
            key = f"{eng} ({mode})"
            res = benchmark_engine(eng, mode)
            if res:
                results[key] = res

    # Print Summary Table
    print("\n" + "="*60)
    print(f"{'Configuration':<30} | {'Time (s)':<10} | {'Entropy':<10}")
    print("-" * 60)
    for key, res in results.items():
        print(f"{key:<30} | {res['time']:<10.4f} | {res['entropy']:<10.4f}")
    print("="*60)
    
    # Analysis
    print("\n🧐 Analysis:")
    ionq_res = results.get("Cartesian/Native (ionq)")
    classic_res = results.get("Cartesian/Native (complex_noise)")
    
    if ionq_res and classic_res:
        time_diff = ionq_res['time'] - classic_res['time']
        print(f"- Latency Cost: Quantum Genesis takes {time_diff:.2f}s longer.")
        
        if ionq_res['entropy'] > classic_res['entropy']:
            print(f"- Complexity Gain: Quantum state has higher entropy (+{ionq_res['entropy'] - classic_res['entropy']:.2f}).")
        else:
            print(f"- Complexity: Entropy is similar or lower (depends on circuit depth).")

if __name__ == "__main__":
    main()
