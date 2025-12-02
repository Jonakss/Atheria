import sys
import os
import torch
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.batch_inference_engine import BatchInferenceEngine
from src import config as cfg

# Configure logging
logging.basicConfig(level=logging.INFO)

def verify_live_ionq():
    print("⚛️ Verifying LIVE IonQ Multiverse Initialization...")
    print(f"   Backend: {cfg.IONQ_BACKEND_NAME}")
    
    if not cfg.IONQ_API_KEY:
        print("❌ Error: IONQ_API_KEY not found in environment.")
        return

    # Setup Engine with a dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x): return x
        
    grid_size = 32
    d_state = 4
    device = torch.device('cpu')
    model = DummyModel()
    
    engine = BatchInferenceEngine(model, grid_size, d_state, device)
    
    # Run Initialization LIVE
    # Requesting 4 universes = 4 shots
    num_universes = 4
    
    try:
        print(f"   🚀 Submitting job to IonQ for {num_universes} universes...")
        engine.initialize_from_ionq_multiverse(num_universes)
        
        print(f"   ✅ Success! Created {len(engine.states)} universes.")
        
        # Verify divergence
        u0 = engine.states[0].psi
        u1 = engine.states[1].psi
        
        diff = torch.sum(torch.abs(u0 - u1)).item()
        print(f"   - Divergence between U0 and U1: {diff:.6f}")
        
        if diff > 0:
            print("   ✨ Universes are distinct (Quantum Randomness confirmed).")
        else:
            print("   ⚠️ Universes are identical (Possible if shots collapsed to same state, or fallback used).")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    verify_live_ionq()
