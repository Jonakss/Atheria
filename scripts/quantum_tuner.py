import sys
import os
import torch
import numpy as np
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.qca_engine import CartesianEngine, QuantumState
from src import config as cfg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuantumTuner:
    def __init__(self, grid_size=32, d_state=4, max_iter=20):
        self.grid_size = grid_size
        self.d_state = d_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_iter = max_iter
        
        # Load a dummy model (or real one if available)
        # For tuning, we want to see how the INITIAL state evolves under the standard physics
        # We can use a simple identity or diffusion model if no trained model is loaded
        # But ideally we use the trained model to see interaction
        self.model = self._load_model()
        
    def _load_model(self):
        # Try to load latest experiment model
        try:
            from src.utils import load_experiment_config, get_latest_checkpoint
            from src.model_loader import load_model
            
            exp_name = cfg.EXPERIMENT_NAME
            config = load_experiment_config(exp_name)
            checkpoint = get_latest_checkpoint(exp_name)
            
            if config and checkpoint:
                model, _ = load_model(config, checkpoint)
                model = model.to(self.device)
                logging.info(f"✅ Loaded model from {exp_name}")
                return model
        except Exception as e:
            logging.warning(f"⚠️ Could not load trained model: {e}. Using Dummy Model.")
            
        # Fallback Dummy
        class DummyModel(torch.nn.Module):
            def forward(self, x): return x * 0.99 # Slight decay
        return DummyModel().to(self.device)

    def objective_function(self, params):
        """
        Evaluates the 'Complexity' of a simulation seeded with 'params'.
        Metric = Entropy * Stability
        """
        # 1. Generate Variational State
        try:
            psi_init = QuantumState.create_variational_state(
                self.grid_size, self.d_state, self.device, params, strength=0.5
            )
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return 0.0 # Bad score
            
        # 2. Run Simulation
        engine = CartesianEngine(
            self.model, self.grid_size, self.d_state, self.device, 
            precomputed_state=psi_init
        )
        
        steps = 20
        entropies = []
        energies = []
        
        for _ in range(steps):
            engine.evolve_internal_state()
            psi = engine.state.psi
            
            # Calculate Entropy
            prob = psi.abs().pow(2)
            prob = prob / (prob.sum() + 1e-9)
            entropy = -torch.sum(prob * torch.log(prob + 1e-9)).item()
            entropies.append(entropy)
            
            # Calculate Energy (Stability proxy)
            energy = torch.sum(psi.abs().pow(2)).item()
            energies.append(energy)
            
        # 3. Calculate Metric
        avg_entropy = np.mean(entropies)
        stability = 1.0 / (np.std(energies) + 1e-5) # Inverse of variance
        
        # Normalize stability to avoid explosion
        stability = min(stability, 10.0)
        
        score = avg_entropy * stability
        logging.info(f"   📊 Score: {score:.4f} (Entropy: {avg_entropy:.2f}, Stability: {stability:.2f})")
        
        return -score # Minimize negative score

    def tune(self):
        logging.info("🎛️ Starting Quantum Tuner (SPSA)...")
        
        # SPSA Parameters
        n_params = 11 # One per qubit
        theta = np.random.rand(n_params) * 2 * np.pi # Initial random angles
        
        alpha = 0.602
        gamma = 0.101
        a = 0.16
        c = 0.1
        A = self.max_iter * 0.1
        
        best_score = float('inf')
        best_theta = theta.copy()
        
        for k in range(self.max_iter):
            ak = a / (k + 1 + A)**alpha
            ck = c / (k + 1)**gamma
            
            delta = np.sign(np.random.rand(n_params) - 0.5)
            
            theta_plus = theta + ck * delta
            theta_minus = theta - ck * delta
            
            y_plus = self.objective_function(theta_plus)
            y_minus = self.objective_function(theta_minus)
            
            grad = (y_plus - y_minus) / (2 * ck * delta)
            
            theta = theta - ak * grad
            
            # Keep track of best
            current_score = (y_plus + y_minus) / 2
            if current_score < best_score:
                best_score = current_score
                best_theta = theta.copy()
                logging.info(f"   🌟 New Best Score: {-best_score:.4f} at iter {k}")
                
        logging.info("✅ Tuning Complete.")
        logging.info(f"   🏆 Best Score: {-best_score:.4f}")
        logging.info(f"   Best Params: {best_theta.tolist()}")
        
        # Save results
        results = {
            "best_score": -best_score,
            "best_params": best_theta.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        with open("best_quantum_params.json", "w") as f:
            json.dump(results, f, indent=2)
            
if __name__ == "__main__":
    tuner = QuantumTuner(max_iter=5) # Short run for testing
    tuner.tune()
