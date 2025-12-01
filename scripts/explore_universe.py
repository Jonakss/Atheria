import numpy as np
import json
import logging
import time
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniverseExplorer")

@dataclass
class UniversePhenotype:
    entropy: float
    stability: float
    fitness: float

@dataclass
class UniverseGenotype:
    gamma_decay: float
    noise_level: float

class QuantumSampler:
    """
    Uses a Quantum Circuit to generate diverse initial parameters.
    """
    def __init__(self, num_params=2):
        self.num_qubits = num_params
        # EfficientSU2 provides a good heuristic for randomizing states
        self.ansatz = EfficientSU2(self.num_qubits, reps=1, entanglement='full')
        self.estimator = StatevectorEstimator()

    def sample(self) -> UniverseGenotype:
        """
        Generates a new set of parameters using Quantum Measurement.
        """
        # Randomize the circuit angles (Quantum Seeding)
        random_angles = np.random.random(self.ansatz.num_parameters) * 2 * np.pi

        # We define two observables to map to our 2 parameters
        # Z on qubit 0 -> gamma_decay
        # Z on qubit 1 -> noise_level
        obs_gamma = SparsePauliOp.from_list([("ZI", 1)]) # Z on q0
        obs_noise = SparsePauliOp.from_list([("IZ", 1)]) # Z on q1

        # Run Estimator
        # Pubs: [(Circuit, Obs1, Params), (Circuit, Obs2, Params)]
        pub1 = (self.ansatz, obs_gamma, random_angles)
        pub2 = (self.ansatz, obs_noise, random_angles)

        job = self.estimator.run([pub1, pub2])
        result = job.result()

        val_gamma = result[0].data.evs
        val_noise = result[1].data.evs

        # Map [-1, 1] to physical ranges
        # gamma_decay: [0.0, 0.1]
        # noise_level: [0.0, 0.5]

        gamma = (float(val_gamma) + 1) / 2 * 0.1
        noise = (float(val_noise) + 1) / 2 * 0.5

        return UniverseGenotype(gamma_decay=gamma, noise_level=noise)

class MockAetheria:
    """
    Simulates the Universe physics to return Behavior Metrics.
    """
    @staticmethod
    def run_simulation(params: UniverseGenotype) -> UniversePhenotype:
        # 1. Spatial Entropy (H_S)
        # Higher noise -> Higher Entropy
        # Higher gamma (decay) -> Lower Entropy (everything dies)
        entropy = 0.5 + params.noise_level - (params.gamma_decay * 5)
        entropy = np.clip(entropy, 0.0, 1.0)

        # 2. Temporal Stability (Delta M)
        # High noise -> Low Stability
        # Low gamma -> High Stability (Conservation)
        stability = 1.0 - params.noise_level - (params.gamma_decay * 2)
        stability = np.clip(stability, 0.0, 1.0)

        # 3. Fitness (Interesancia)
        # We want "Edge of Chaos": Medium Entropy, High Stability
        # Gaussian peak around Entropy=0.5
        fitness = np.exp(-((entropy - 0.5)**2) / 0.1) * stability

        # Add random jitter
        fitness += np.random.normal(0, 0.01)

        return UniversePhenotype(entropy, stability, fitness)

class MAPElitesArchive:
    """
    The Archive grid storage.
    """
    def __init__(self, resolution=10):
        self.resolution = resolution
        self.grid: Dict[str, dict] = {} # Key: "x_y", Value: {genotype, phenotype}

    def _get_index(self, phenotype: UniversePhenotype) -> str:
        # Discretize 0..1 into resolution bins
        x = int(phenotype.entropy * self.resolution)
        y = int(phenotype.stability * self.resolution)
        # Clamp
        x = min(x, self.resolution - 1)
        y = min(y, self.resolution - 1)
        return f"{x}_{y}"

    def add(self, genotype: UniverseGenotype, phenotype: UniversePhenotype) -> bool:
        idx = self._get_index(phenotype)

        # If cell is empty or new individual is better
        if idx not in self.grid or phenotype.fitness > self.grid[idx]['fitness']:
            self.grid[idx] = {
                'genotype': asdict(genotype),
                'phenotype': asdict(phenotype),
                'fitness': phenotype.fitness
            }
            return True # Added/Replaced
        return False

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.grid, f, indent=2)

def main():
    logger.info("Initializing THE EXPLORER (MAP-Elites + Quantum)...")

    sampler = QuantumSampler(num_params=2)
    archive = MAPElitesArchive(resolution=10) # 10x10 Grid

    total_generations = 100

    start_time = time.time()

    for i in range(total_generations):
        # 1. Generate (Quantum Sampling)
        genotype = sampler.sample()

        # 2. Evaluate (Mock Simulation)
        phenotype = MockAetheria.run_simulation(genotype)

        # 3. Update Archive
        added = archive.add(genotype, phenotype)

        if added:
            logger.info(f"Gen {i}: New Elite found! Ent={phenotype.entropy:.2f}, Stab={phenotype.stability:.2f}, Fit={phenotype.fitness:.2f}")

    # Save Results
    output_path = "output/universe_atlas.json"
    # Ensure output dir exists
    os.makedirs("output", exist_ok=True)

    archive.save(output_path)

    elapsed = time.time() - start_time
    logger.info(f"Exploration Complete. {len(archive.grid)} niches filled out of {archive.resolution**2} possible.")
    logger.info(f"Atlas saved to {output_path}")

if __name__ == "__main__":
    main()
