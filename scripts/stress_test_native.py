import time
import torch
import argparse
import logging
import sys
import psutil
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.engines.native_engine_wrapper import NativeEngineWrapper
from src.engines.qca_engine import Aetheria_Motor, QuantumState
from src.utils import get_latest_jit_model

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def create_dummy_model(d_state, device):
    """Creates a dummy TorchScript model for testing."""
    class DummyModel(torch.nn.Module):
        def __init__(self, d_state):
            super().__init__()
            self.conv = torch.nn.Conv2d(2*d_state, 2*d_state, kernel_size=3, padding=1)
        
        def forward(self, x):
            return self.conv(x)

    model = DummyModel(d_state).to(device)
    example_input = torch.randn(1, 2*d_state, 20, 20).to(device) # Chunk size
    traced_model = torch.jit.trace(model, example_input)
    return traced_model

def run_stress_test(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Stress Test")
    print(f"   Engine: {args.engine.upper()}")
    print(f"   Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"   Density: {args.density*100}%")
    print(f"   Device: {device}")
    
    d_state = 8
    
    # Create Dummy Model
    model_path = "stress_test_model.pt"
    if not os.path.exists(model_path):
        print("   🔨 Creating dummy model...")
        traced_model = create_dummy_model(d_state, device)
        traced_model.save(model_path)
    
    # Initialize Engine
    print("   ⚙️ Initializing Engine...")
    start_mem = get_memory_usage()
    
    if args.engine == 'native':
        # Native Engine (Sparse)
        # We need to manually initialize it to control density
        # NativeEngineWrapper expects a config object usually, but we can hack it
        from types import SimpleNamespace
        cfg = SimpleNamespace(INITIAL_STATE_MODE_INFERENCE='complex_noise')
        
        engine = NativeEngineWrapper(grid_size=args.grid_size, d_state=d_state, device=args.device, cfg=cfg)
        engine.load_model(model_path)
        
        # Inject noise manually to control density
        print(f"   🌱 Injecting noise (Density: {args.density})...")
        # Generate random coordinates
        num_particles = int(args.grid_size * args.grid_size * args.density)
        
        # We can use the wrapper's internal state to set this up, but it's easier to just add particles
        # However, adding 1M particles one by one is slow.
        # Let's use a dense state and convert it.
        
        # Create a dense state with noise (B, H, W, C)
        state = torch.zeros(1, args.grid_size, args.grid_size, d_state, dtype=torch.complex64, device=device)
        
        # Random indices
        indices = torch.randperm(args.grid_size * args.grid_size, device=device)[:num_particles]
        
        # Fill with noise
        noise = torch.randn(num_particles, d_state, dtype=torch.complex64, device=device)
        
        # Flatten and scatter
        state_flat = state.view(1, -1, d_state)
        # We need to scatter noise into state_flat at indices
        mask = torch.zeros(args.grid_size * args.grid_size, device=device, dtype=torch.bool)
        mask[indices] = True
        mask = mask.view(1, args.grid_size, args.grid_size, 1)
        
        noise_grid = torch.randn(1, args.grid_size, args.grid_size, d_state, dtype=torch.complex64, device=device)
        state = torch.where(mask, noise_grid, state)
        
        # Initialize native from dense
        engine._initialize_native_state_from_dense(state)
        
    else:
        # Python Engine (Dense)
        model = torch.jit.load(model_path).to(device)
        engine = Aetheria_Motor(model_operator=model, grid_size=args.grid_size, d_state=d_state, device=args.device)
        
        # Inject noise
        print(f"   🌱 Injecting noise (Density: {args.density})...")
        num_particles = int(args.grid_size * args.grid_size * args.density)
        indices = torch.randperm(args.grid_size * args.grid_size, device=device)[:num_particles]
        mask = torch.zeros(args.grid_size * args.grid_size, device=device, dtype=torch.bool)
        mask[indices] = True
        mask = mask.view(1, 1, args.grid_size, args.grid_size)
        noise_grid = torch.randn(1, d_state, args.grid_size, args.grid_size, dtype=torch.complex64, device=device)
        engine.state.psi = torch.where(mask, noise_grid, engine.state.psi)

    init_mem = get_memory_usage()
    print(f"   💾 Memory Usage: {init_mem - start_mem:.2f} MB (Engine + State)")
    
    if args.engine == 'native':
        print(f"   🧩 Active Particles: {engine.native_engine.get_matter_count()}")
    
    # Warmup
    print("   🔥 Warming up...")
    for _ in range(5):
        if args.engine == 'native':
            engine.evolve_internal_state()
        else:
            engine.step()
            
    # Benchmark
    print(f"   🏃 Running {args.steps} steps...")
    start_time = time.time()
    
    for i in range(args.steps):
        if args.engine == 'native':
            engine.evolve_internal_state()
        else:
            engine.step()
            
        if (i+1) % 10 == 0:
            print(f"      Step {i+1}/{args.steps}...", end='\r')
            
    end_time = time.time()
    duration = end_time - start_time
    sps = args.steps / duration
    
    print(f"\n   📊 Results:")
    print(f"      Time: {duration:.4f} s")
    print(f"      SPS:  {sps:.2f} steps/sec")
    print(f"      MSP:  {(duration/args.steps)*1000:.2f} ms/step")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test Atheria engines")
    parser.add_argument('--engine', type=str, choices=['native', 'python'], required=True, help="Engine to test")
    parser.add_argument('--grid_size', type=int, default=512, help="Grid size (NxN)")
    parser.add_argument('--density', type=float, default=0.1, help="Initial particle density (0.0 - 1.0)")
    parser.add_argument('--steps', type=int, default=50, help="Number of steps to run")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    run_stress_test(args)
