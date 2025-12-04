import os
import sys
import torch
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qca_engine_pennylane import QuantumCell

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [MODEL_SELECTOR] - %(message)s')

def evaluate_model(universe_id, params, dev_name="ionq.simulator"):
    """
    Evaluates a single model configuration (Universe) on the Quantum Backend.
    """
    logging.info(f"🌌 Evaluating Universe #{universe_id} | Params: {params}")

    # Initialize a specific cell configuration
    n_qubits = 4
    q_cell = QuantumCell(n_qubits=n_qubits, n_layers=params['layers'], dev_name=dev_name)

    # Create input state (the 'Superposition' probe)
    # We use a fixed input to compare how different 'laws of physics' (weights) affect it
    input_state = torch.tensor([np.pi/2] * n_qubits).unsqueeze(0) # |+> state roughly

    # Run the model
    with torch.no_grad():
        probs = q_cell(input_state)

    # Calculate a metric: Entropy (Complexity)
    # H = -sum(p * log(p))
    probs_np = probs.numpy().flatten()
    entropy = -np.sum(probs_np * np.log(probs_np + 1e-9))

    return {
        "id": universe_id,
        "params": params,
        "entropy": entropy,
        "output_dist": probs_np[:4] # Store partial distribution for log
    }

def main():
    print("\n🔭 MULTIVERSE MODEL SELECTOR (IONQ) 🔭")
    print("Searching for the 'Best' Model from Superposition...\n")

    api_key = os.getenv("IONQ_API_KEY")
    dev_name = "ionq.simulator"
    if not api_key:
        print("⚠️ IONQ_API_KEY not found. Using local simulation.")
        dev_name = "default.qubit"

    # Define the 'Superposition' of Models
    # These represent different hyperparameters we want to test
    universes = [
        {"layers": 1},
        {"layers": 2},
        {"layers": 3},
        {"layers": 1} # Conservative universe
    ]

    results = []

    print(f"🚀 Launching {len(universes)} universes on {dev_name}...\n")

    for i, params in enumerate(universes):
        try:
            res = evaluate_model(i, params, dev_name)
            results.append(res)
            print(f"✅ Universe #{i}: Entropy = {res['entropy']:.4f}")
        except Exception as e:
            print(f"❌ Universe #{i} collapsed: {e}")

    # Check for empty results
    if not results:
        print("\n❌ All universes failed to evaluate. No best model found.")
        return

    # Find the 'Best' Model
    # Criteria: Maximum Entropy (Maximum Complexity/Interest)
    # Or could be minimum entropy (Stability). Let's pick Max Entropy for 'Life'.
    best_model = max(results, key=lambda x: x['entropy'])

    print("\n🏆 BEST MODEL FOUND FROM SUPERPOSITION 🏆")
    print(f"ID: {best_model['id']}")
    print(f"Parameters: {best_model['params']}")
    print(f"Entropy: {best_model['entropy']:.4f}")
    print(f"Distribution Head: {best_model['output_dist']}")

    print("\n📝 Recommendation: Deploy Universe #{} for production run.".format(best_model['id']))

if __name__ == "__main__":
    main()
