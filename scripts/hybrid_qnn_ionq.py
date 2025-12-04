import os
import sys
import torch
import numpy as np
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qca_engine_pennylane import QuantumKernel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [HYBRID_QNN] - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Hybrid Quantum Neural Network (IonQ) Inference")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--height", type=int, default=5, help="Input image height")
    parser.add_argument("--width", type=int, default=5, help="Input image width")
    parser.add_argument("--device", type=str, default="ionq.simulator", help="PennyLane device name (e.g., ionq.simulator, default.qubit)")

    args = parser.parse_args()

    print("\n🧠 HYBRID QUANTUM NEURAL NETWORK (IONQ) 🧠\n")

    # Check for API Key if using IonQ
    if "ionq" in args.device:
        api_key = os.getenv("IONQ_API_KEY")
        if not api_key:
            print("⚠️ IONQ_API_KEY not found. This will fall back to local simulator.")
            print("   -> export IONQ_API_KEY='your_key_here'")

    print(f"🔧 Configuration: {args.height}x{args.width} Input, Device: {args.device}")

    # 1. Initialize Hybrid Model
    # We use a small kernel to avoid massive qubit usage if running on real hardware/simulator quotas
    try:
        q_kernel = QuantumKernel(n_qubits=9, n_layers=1, dev_name=args.device)
        print("✅ Hybrid Quantum Kernel Initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Quantum Kernel: {e}")
        return

    # 2. Create Dummy Input (Random Noise / 'Vacuum State')
    input_data = torch.rand(args.batch_size, args.channels, args.height, args.width) * np.pi # Encode as angles [0, pi]
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
        if output.shape[2] >= 3 and output.shape[3] >= 3:
             print(output[0, 0, :3, :3])
        else:
             print(output)

    except Exception as e:
        print(f"❌ Inference Failed: {e}")

if __name__ == "__main__":
    main()
