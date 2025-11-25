import torch
import sys
import os
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.engines.native_engine_wrapper import NativeEngineWrapper

class MockModel(torch.nn.Module):
    def __init__(self, d_state):
        super().__init__()
        self.conv = torch.nn.Conv2d(d_state * 2, d_state * 2, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x is [batch, 2*d_state, H, W]
        # Simple diffusion + growth
        return self.conv(x) * 0.9 + 0.05

def test_native_infinite_universe():
    print("🌌 Testing Native Engine Infinite Universe...")
    
    device = "cpu"
    d_state = 8
    grid_size = 256
    
    # 1. Create and Save Dummy Model
    print("   🔨 Creating dummy TorchScript model...")
    model = MockModel(d_state)
    model.eval()
    # Trace model with example input
    # Input shape: [1, 2*d_state, grid_size, grid_size] (as expected by Native Engine build_batch_input)
    # Wait, build_batch_input uses patch_size = grid_size.
    example_input = torch.randn(1, d_state * 2, grid_size, grid_size)
    traced_model = torch.jit.trace(model, example_input)
    
    model_path = "dummy_native_model.pt"
    traced_model.save(model_path)
    print(f"   ✅ Model saved to {model_path}")
    
    try:
        # 2. Initialize Engine
        print("   ⚙️ Initializing NativeEngineWrapper...")
        engine = NativeEngineWrapper(grid_size, d_state, device)
        
        # 3. Load Model
        print("   📥 Loading model...")
        success = engine.load_model(model_path)
        if not success:
            print("   ❌ Failed to load model!")
            return
        print("   ✅ Model loaded")
        
        # 4. Inject Genesis Seed (via dense state)
        print("   🌱 Injecting Genesis Seed...")
        # NativeEngineWrapper initializes from self.state.psi
        # We can manually inject into native_engine
        
        # Create a seed particle
        seed_state = torch.randn(d_state, dtype=torch.complex64)
        # Center of grid
        cx, cy = grid_size // 2, grid_size // 2
        
        # Use internal native engine method if available, or wrapper helper
        # Wrapper doesn't have add_particle exposed directly, but we can access native_engine
        
        # Let's use the wrapper's state and re-initialize
        engine.state.psi = torch.zeros(1, grid_size, grid_size, d_state, dtype=torch.complex64)
        engine.state.psi[0, cy, cx] = seed_state
        
        # Clear existing matter (from default initialization)
        engine.native_engine.clear()
        
        # Force re-initialization from dense
        engine._initialize_native_state_from_dense(engine.state.psi)
        
        count = engine.native_engine.get_matter_count()
        print(f"   Initial matter count: {count}")
        assert count > 0, "Failed to inject matter"
        
        # 5. Run Simulation Loop
        print("   🚀 Running simulation loop (10 steps)...")
        start_time = time.time()
        
        for i in range(10):
            engine.evolve_internal_state()
            count = engine.native_engine.get_matter_count()
            print(f"   Step {i+1}: {count} particles")
            
        end_time = time.time()
        print(f"   ✅ Simulation finished in {end_time - start_time:.2f}s")
        
        # 6. Verify Expansion
        assert count > 1, "Universe did not expand!"
        print("   ✅ Universe expansion confirmed")
        
        # 7. Verify Viewport Generation (Dense State)
        print("   🔭 Generating viewport (get_dense_state)...")
        viewport = engine.get_dense_state()
        print(f"   Viewport shape: {viewport.shape}")
        
        # Check if viewport has data
        viewport_abs = viewport.abs()
        max_val = viewport_abs.max().item()
        print(f"   Viewport max value: {max_val}")
        assert max_val > 0, "Viewport is empty!"
        
        print("   ✅ Viewport generated successfully")
        
    finally:
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)
            print("   🧹 Cleaned up model file")

if __name__ == "__main__":
    test_native_infinite_universe()
