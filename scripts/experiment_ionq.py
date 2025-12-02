#!/usr/bin/env python3
"""
IonQ Experiment Script for Atheria
==================================

This script demonstrates how to run an Atheria experiment using the IonQ backend.
It initializes the engine with the IonQBackend and runs a simple quantum circuit execution.

Usage:
    export IONQ_API_KEY="your_api_key"
    python scripts/experiment_ionq.py
"""

import os
import sys
import logging
import torch
from qiskit import QuantumCircuit

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.compute_backend import IonQBackend
from src import config

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 Starting IonQ Experiment for Atheria...")
    
    # Check for API Key
    api_key = config.IONQ_API_KEY
    if not api_key:
        print("⚠️  No IONQ_API_KEY found in environment variables or config.")
        print("   You can set it via: export IONQ_API_KEY='your_key'")
        print("   Proceeding will likely fail or run in mock mode if implemented.")
    
    # Initialize Backend
    try:
        backend = IonQBackend(api_key=api_key, backend_name=config.IONQ_BACKEND_NAME)
        status = backend.get_status()
        print(f"✅ Backend Initialized: {status}")
        
        if status['status'] == 'offline' and not api_key:
             print("❌ Backend is offline (likely due to missing API key). Exiting.")
             return

    except Exception as e:
        print(f"❌ Failed to initialize backend: {e}")
        return

    # Create a simple Quantum Circuit (Bell State)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    print("\n🧪 Submitting Circuit to IonQ...")
    print(qc)
    
    try:
        # Execute
        counts = backend.execute('run_circuit', qc, shots=100)
        print("\n🏆 Results:")
        print(counts)
        
        # Simple validation
        if '00' in counts and '11' in counts:
            print("✅ Bell state correlation observed!")
        else:
            print("⚠️  Unexpected results (might be noise or simulator artifact).")
            
    except Exception as e:
        print(f"❌ Execution failed: {e}")

if __name__ == "__main__":
    main()
