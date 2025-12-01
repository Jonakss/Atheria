import sys
import os
sys.path.insert(0, os.getcwd())

import time
import torch
import psutil
import numpy as np
from src.engines.native_engine_wrapper import NativeEngineWrapper

def run_benchmark(num_threads, steps=20, grid_size=128, d_state=64):
    print(f"\n--- Benchmarking with {num_threads} threads ---")
    
    # Set OpenMP threads
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    
    # Initialize Engine
    try:
        engine = NativeEngineWrapper(grid_size=grid_size, d_state=d_state, device="cpu")
        
        # Load dummy model (ensure it exists)
        model_path = "dummy_native_model_128ch.pt"
        if not os.path.exists(model_path):
            print(f"Error: Model {model_path} not found. Run generate_dummy_model.py first.")
            return None

        if not engine.load_model(model_path):
            print("Error: Failed to load model.")
            return None
            
        # Add initial particles
        num_particles = int(grid_size * grid_size * 0.05) # 5% fill
        engine.add_initial_particles(num_particles)
        
        # Warmup
        print("Warming up...")
        for i in range(5):
            print(f"Warmup step {i+1}/5")
            engine.evolve_internal_state()
            
        # Benchmark
        print(f"Running {steps} steps...")
        start_time = time.time()
        for i in range(steps):
            if i % 10 == 0:
                print(f"Step {i}/{steps}")
            engine.evolve_internal_state()
        end_time = time.time()
        
        duration = end_time - start_time
        fps = steps / duration
        
        print(f"Completed in {duration:.4f}s")
        print(f"FPS: {fps:.2f}")
        
        engine.cleanup()
        return fps
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None

def main():
    # Generate dummy model if needed
    if not os.path.exists("dummy_native_model_128ch.pt"):
        print("Generating dummy model...")
        os.system("python3 scripts/generate_dummy_model.py")

    thread_counts = [1, 2, 4, 6, 8, 12, 16]
    results = {}
    
    print("Starting Parallelism Benchmark...")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    for t in thread_counts:
        fps = run_benchmark(t)
        if fps is not None:
            results[t] = fps
            
    print("\n--- Final Results ---")
    print(f"{'Threads':<10} | {'FPS':<10} | {'Speedup (vs 1)':<15}")
    print("-" * 40)
    
    base_fps = results.get(1, 1.0)
    
    for t in thread_counts:
        if t in results:
            fps = results[t]
            speedup = fps / base_fps
            print(f"{t:<10} | {fps:<10.2f} | {speedup:<15.2f}x")
            
    best_threads = max(results, key=results.get)
    print(f"\nOptimal Thread Count: {best_threads} (FPS: {results[best_threads]:.2f})")

if __name__ == "__main__":
    main()
