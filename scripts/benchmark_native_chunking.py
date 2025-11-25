import time
import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.engines.native_engine_wrapper import NativeEngineWrapper

class MockModel(torch.nn.Module):
    def __init__(self, d_state):
        super().__init__()
        self.d_state = d_state
        # Simple conv to simulate work
        self.conv = torch.nn.Conv2d(2*d_state, 2*d_state, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x)

def benchmark_chunking():
    print("🚀 Benchmarking Native Engine Chunking...")
    
    grid_size = 256
    d_state = 8
    device = "cpu"
    
    # 1. Create Model
    model = MockModel(d_state)
    model.eval()
    # Chunk size 16 + padding 2 -> input size 20
    example_input = torch.randn(1, 2*d_state, 20, 20)
    traced_model = torch.jit.trace(model, example_input)
    model_path = "dummy_chunk_model.pt"
    traced_model.save(model_path)
    
    # 2. Init Engine
    engine = NativeEngineWrapper(grid_size, d_state, device)
    engine.load_model(model_path)
    
    # 3. Inject Seeds (create sparse activity)
    # Inject 5 seeds far apart to activate 5 independent chunks
    print("   🌱 Injecting 5 seeds...")
    engine.native_engine.clear()
    
    seeds = [
        (64, 64), (192, 64), (128, 128), (64, 192), (192, 192)
    ]
    
    for x, y in seeds:
        seed_state = torch.randn(d_state, dtype=torch.complex64)
        # We need to use the internal method or wrapper helper
        # Wrapper initializes from state.psi, let's just hack it via native_engine directly if exposed
        # But add_particle expects Coord3D which is C++ type.
        # Let's use the wrapper's state re-init trick
        pass

    # Actually, let's just use the wrapper's _initialize_native_state_from_dense
    # Create a dense state with these 5 seeds
    engine.state.psi = torch.zeros(1, grid_size, grid_size, d_state, dtype=torch.complex64)
    for x, y in seeds:
         engine.state.psi[0, y, x] = torch.randn(d_state, dtype=torch.complex64)
    
    engine._initialize_native_state_from_dense(engine.state.psi)
    
    initial_count = engine.native_engine.get_matter_count()
    print(f"   Initial particles: {initial_count}")
    
    # 4. Run Benchmark
    steps = 10
    print(f"   🏃 Running {steps} steps...")
    
    start_time = time.time()
    for i in range(steps):
        engine.evolve_internal_state()
    end_time = time.time()
    
    total_time = end_time - start_time
    sps = steps / total_time
    
    final_count = engine.native_engine.get_matter_count()
    print(f"   ✅ Finished.")
    print(f"   Final particles: {final_count}")
    print(f"   Time: {total_time:.4f}s")
    print(f"   Speed: {sps:.2f} steps/sec")

if __name__ == "__main__":
    benchmark_chunking()
