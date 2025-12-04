import os
import sys
import torch
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qca_engine_pennylane import QuantumKernel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [HYBRID_QNN] - %(message)s')

def main():
    print("\n🧠 HYBRID QUANTUM NEURAL NETWORK (IONQ) 🧠\n")

    # Check for API Key
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        print("⚠️ IONQ_API_KEY not found. This will fall back to local simulator.")
        print("   -> export IONQ_API_KEY='your_key_here'")

    # Configuration
    BATCH_SIZE = 1
    CHANNELS = 1
    HEIGHT = 5
    WIDTH = 5
    DEV_NAME = "ionq.simulator"

    print(f"🔧 Configuration: {HEIGHT}x{WIDTH} Input, Device: {DEV_NAME}")

    # 1. Initialize Hybrid Model
    # We use a small kernel to avoid massive qubit usage if running on real hardware/simulator quotas
    try:
        q_kernel = QuantumKernel(n_qubits=9, n_layers=1, dev_name=DEV_NAME)
        print("✅ Hybrid Quantum Kernel Initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Quantum Kernel: {e}")
        return

    # 2. Create Dummy Input (Random Noise / 'Vacuum State')
    input_data = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH) * np.pi # Encode as angles [0, pi]
    print(f"📉 Input Tensor Shape: {input_data.shape}")

    # 3. Time Forward (Inference Pass)
    print("🚀 Running Time Forward (Inference)...")
    try:
        # Move to CPU for this script
        with torch.no_grad():
            output = q_kernel(input_data)

        print("✅ Inference Complete!")
        print(f"📈 Output Tensor Shape: {output.shape}")
        print("\n📊 Output Sample (First 3x3 block):")
        print(output[0, 0, :3, :3])

    except Exception as e:
        print(f"❌ Inference Failed: {e}")

if __name__ == "__main__":
    main()
