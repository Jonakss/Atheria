import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import correctly
from src.engines.qca_engine import Aetheria_Motor
from src.engines.native_engine_wrapper import NativeEngineWrapper

class MockModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.dirac_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def forward(self, x):
        return self.conv(x)

def benchmark_engine(engine_type, name, steps=500, warmup=50):
    print(f"\n--- Benchmarking {name} ---")
    
    grid_size = 128
    d_state = 64  # 128 channels
    device = torch.device("cpu")
    
    # Measure initialization time
    start_init = time.perf_counter()
    
    if engine_type == "native":
        # Native engine needs model path
        model_path = "dummy_native_model_128ch.pt"
        if not os.path.exists(model_path):
            print(f"Error: Model {model_path} not found.")
            return None
        
        # Native wrapper expects arguments directly
        # Pass device as string
        engine = NativeEngineWrapper(grid_size=grid_size, d_state=d_state, device=str(device))
        if not engine.load_model(model_path):
            print("Failed to load model in Native Engine")
            return None
        
    else:
        # Python engine
        model = MockModel(channels=d_state*2)
        # Aetheria_Motor init: model, grid_size, d_state, device (object)
        engine = Aetheria_Motor(model, grid_size, d_state, device)
        pass

    # State is initialized in __init__ for both engines
    # engine.initialize_state(mode="random")
    end_init = time.perf_counter()
    init_time = end_init - start_init
    print(f"Initialization Time: {init_time:.4f} s")
    
    # Warmup
    print(f"Warming up ({warmup} steps)...")
    for i in range(warmup):
        engine.evolve_internal_state(step=i)
        # Add periodic yield to prevent blocking and allow signal handlers
        if (i+1) % 10 == 0:
            time.sleep(0.001)  # 1ms yield
            print(f"  Warmup: {i+1}/{warmup}")
        
    # Benchmark
    print(f"Running benchmark ({steps} steps)...")
    start_bench = time.perf_counter()
    for i in range(steps):
        engine.evolve_internal_state(step=i+warmup)
            
        if (i+1) % 100 == 0:
            print(f"  Step {i+1}/{steps}")
            
    end_bench = time.perf_counter()
    total_time = end_bench - start_bench
    fps = steps / total_time
    
    print(f"Total Time: {total_time:.4f} s")
    print(f"Average FPS: {fps:.2f}")
    
    return {
        "name": name,
        "init_time": init_time,
        "fps": fps
    }

def main():
    print("Starting Comprehensive Benchmark (Python vs C++ Native)...")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not Set')}")
    
    # Ensure dummy model exists
    if not os.path.exists("dummy_native_model_128ch.pt"):
        print("Generating dummy model...")
        import subprocess
        subprocess.run(["python3", "scripts/generate_dummy_model.py"], check=True)

    results = []
    
    # Benchmark Python Engine
    res_py = benchmark_engine("python", "Python Engine (PyTorch)", steps=100, warmup=10)
    if res_py:
        results.append(res_py)
        
    # Benchmark Native Engine (reduced steps for faster completion)
    res_cpp = benchmark_engine("native", "Native Engine (C++)", steps=200, warmup=20)
    if res_cpp:
        results.append(res_cpp)
        
    # Comparison
    if len(results) == 2:
        py_fps = results[0]["fps"]
        cpp_fps = results[1]["fps"]
        speedup = cpp_fps / py_fps
        
        print("\n=== RESULTS ===")
        print(f"{'Engine':<25} | {'FPS':<10} | {'Init (s)':<10}")
        print("-" * 50)
        for res in results:
            print(f"{res['name']:<25} | {res['fps']:<10.2f} | {res['init_time']:<10.4f}")
        print("-" * 50)
        print(f"Speedup Factor: {speedup:.2f}x")
        
        # Save to file
        with open("benchmark_results.txt", "w") as f:
            f.write(f"Speedup: {speedup:.2f}x\n")
            f.write(f"Python FPS: {py_fps:.2f}\n")
            f.write(f"C++ FPS: {cpp_fps:.2f}\n")

if __name__ == "__main__":
    main()
