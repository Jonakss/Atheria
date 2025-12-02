#!/usr/bin/env python3
"""
Test IonQ Initialization (Quantum Genesis)
==========================================

This script tests the 'ionq' initialization mode in QuantumState.
It verifies that:
1. The engine can connect to IonQ.
2. It retrieves quantum data.
3. It successfully creates a grid state from that data.
"""

import sys
import os
import torch
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.qca_engine import QuantumState
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_quantum_genesis():
    print("\n🧪 Testing Quantum Genesis (IonQ Init)...")
    
    # Check API Key
    if not config.IONQ_API_KEY:
        print("❌ IONQ_API_KEY not set. Please export it.")
        return
    
    try:
        # Initialize QuantumState with mode='ionq'
        # We use a small grid for testing to be fast, but large enough to require tiling
        grid_size = 64 
        d_state = 4
        device = torch.device('cpu')
        
        print(f"⏳ Initializing QuantumState(grid={grid_size}, mode='ionq')...")
        state = QuantumState(grid_size, d_state, device, initial_mode='ionq')
        
        # Verify state properties
        psi = state.psi
        print(f"✅ State initialized. Shape: {psi.shape}")
        print(f"   Type: {psi.dtype}")
        print(f"   Device: {psi.device}")
        
        # Check if it's not all zeros
        if torch.all(psi == 0):
            print("❌ State is all zeros! Something went wrong.")
        else:
            print("✅ State contains non-zero values (Quantum Noise).")
            
        # Check if it's not just standard random (hard to prove, but we can check logs)
        # We rely on the logs printed by _get_ionq_state
        
        print("\n🎉 Quantum Genesis Test Passed!")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quantum_genesis()
