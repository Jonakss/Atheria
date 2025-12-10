
import torch
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.lagrangian_engine import LagrangianEngine
from src.models.lagrangian_net import LagrangianNetwork

def test_lnn_conservation():
    print("=== Testing Lagrangian Neural Network Integrator ===")
    
    # Config
    grid_size = 32
    d_state = 1 # 1D Oscillator per pixel for simplicity check
    dt = 0.1
    steps = 100
    
    cfg = {
        'grid_size': grid_size,
        'd_state': d_state,
        'dt': dt,
        'device': 'cpu'
    }
    
    # Init Engine
    engine = LagrangianEngine(cfg)
    
    # Hack: Force model to be a Harmonic Oscillator L = 0.5*v^2 - 0.5*q^2
    # This means a = -q. Simple Harmonic Motion.
    # To do this, we manually set weights of the conv layers or just mock the forward.
    # But let's see what the random initialization does first. It should drift but run.
    
    print("Initial State Energy (approx):", (0.5 * engine.v**2 + 0.5 * engine.q**2).mean().item())
    
    energies = []
    qs = []
    
    # Run loop
    for i in range(steps):
        engine.evolve_internal_state()
        
        # Calculate naive energy H = T + V
        # Assuming the random net approximates something, let's just track q magnitude
        energy = (0.5 * engine.v**2 + 0.5 * engine.q**2).mean().item()
        energies.append(energy)
        qs.append(engine.q[0, 0, 16, 16].item()) # Track one center pixel
        
        if i % 10 == 0:
            print(f"Step {i}: Energy={energy:.4f}, q_center={qs[-1]:.4f}")

    # Plot
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(energies)
        plt.title('Total "Energy" (0.5v^2 + 0.5q^2)')
        plt.xlabel('Step')
        
        plt.subplot(1, 2, 2)
        plt.plot(qs)
        plt.title('Center Pixel Trajectory')
        plt.xlabel('Step')
        
        plt.tight_layout()
        plt.savefig('output/lnn_test_plot.png')
        print("Plot saved to output/lnn_test_plot.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("Test Complete.")

if __name__ == "__main__":
    test_lnn_conservation()
