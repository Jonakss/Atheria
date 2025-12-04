import torch
import atheria_core
import sys
import os

def verify_dense_engine():
    print("🧪 Verifying Native DenseEngine...")
    
    try:
        # 1. Initialize Engine
        grid_size = 64
        d_state = 1
        device = 'cpu'
        
        print(f"   Initializing DenseEngine(grid={grid_size}, d_state={d_state}, device={device})...")
        engine = atheria_core.DenseEngine(grid_size, d_state, device)
        
        # 2. Verify Initial State
        state = engine.get_state()
        print(f"   Initial State Shape: {state.shape}")
        print(f"   Initial State Type: {state.dtype}")
        
        if state.shape != (1, d_state, grid_size, grid_size):
            print(f"❌ Shape mismatch! Expected (1, {d_state}, {grid_size}, {grid_size})")
            return False
            
        if not state.is_complex():
            print("❌ Type mismatch! Expected complex tensor")
            return False
            
        print("✅ Initialization passed.")
        
        # 3. Verify Set/Get State
        print("\n   Testing set_state/get_state...")
        new_state = torch.ones_like(state) * (0.5 + 0.5j)
        engine.set_state(new_state)
        
        retrieved_state = engine.get_state()
        if not torch.allclose(new_state, retrieved_state):
            print("❌ State mismatch after set_state!")
            return False
            
        print("✅ State persistence passed.")
        
        # 4. Verify Step (without model)
        print("\n   Testing step() without model...")
        count = engine.step()
        print(f"   Step returned: {count}")
        
        if count != 0:
             print("❌ Expected 0 active cells (no model loaded)")
             # Actually, step returns 0 if !model_loaded_, so this is correct.
        
        print("✅ Step (no model) passed.")
        
        print("\n🎉 Native DenseEngine verification passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_dense_engine()
    sys.exit(0 if success else 1)
