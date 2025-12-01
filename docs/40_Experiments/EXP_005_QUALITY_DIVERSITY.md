# EXP-005: Quality Diversity and Quantum Exploration

**Date:** 2024-05-22
**Status:** In Progress
**Objective:** Map the "Atlas of Possibilities" of Aetheria using MAP-Elites and Quantum Diversity Sampling.

## 1. Research: Moving Beyond Optimization

Standard optimization (like VQE or Gradient Descent) converges to a *single* optimal point. However, in Artificial Life, we often want a *diverse set* of interesting behaviors (e.g., stable solitons, chaotic gliders, growing structures).

### A. MAP-Elites (Multi-dimensional Archive of Phenotypic Elites)
**Concept:** Instead of a single "best individual", we maintain a Grid (Archive) of elites. Each cell in the grid corresponds to a specific combination of behavior traits.

**Application to Aetheria:**
*   **Genotype (Inputs):** The Physics Hyperparameters ($\theta$).
    *   $\gamma$ (`gamma_decay`): Dissipation rate.
    *   $\alpha$ (`learning_rate`): Speed of evolution.
    *   $\sigma$ (`noise_level`): Stochasticity.
*   **Phenotype (Behavior Descriptors):**
    *   **BC1: Spatial Entropy ($H_S$):** Measures the visual complexity of the universe.
        *   Low: Empty space or uniform fill.
        *   High: Complex structures, noise.
    *   **BC2: Temporal Stability ($\Delta M$):** Measures how the total mass changes over time.
        *   Low Variance: Stable/Conservation laws hold.
        *   High Variance: Exploding/Dying universes.
*   **Fitness:** "Interesancia" (e.g., Symmetry $\times$ Survival Rate).

**The Archive:**
We create a 2D grid where X-axis is $H_S$ and Y-axis is $\Delta M$.
If a new parameter set $\theta'$ maps to cell $(x, y)$ and has higher fitness than the current occupant, we replace it.

### B. Quantum Diversity Sampling
**Hypothesis:** Quantum algorithms can sample the parameter space more diversely than uniform random sampling.

**Mechanism:**
*   **Quantum Circuit:** An ansatz with high entanglement capability (e.g., `EfficientSU2` or `RealAmplitudes` with full entanglement).
*   **Measurement:** We measure the state in the computational basis.
*   **Mapping:** The bitstrings (or probability amplitudes) map to regions in the hyperparameter space.
*   **Advantage:** Quantum circuits can generate non-local correlations and explore "rough" landscapes effectively. We use the quantum computer as a **High-Dimensional Seed Generator**.

## 2. Architecture: "The Explorer"

We propose a background script (`scripts/explore_universe.py`) that implements the MAP-Elites loop.

### Algorithm
1.  **Initialize:** Create an empty Archive `Map = {}`.
2.  **Generate Population:**
    *   Use **Quantum Sampling** to generate $N$ initial parameter sets.
3.  **Evaluation Loop:**
    *   For each parameter set $\theta$:
        *   Run Aetheria Simulation ($T=100$ steps).
        *   Calculate Descriptors $(H_S, \Delta M)$ and Fitness $F$.
        *   Locate Cell $C$ in the Archive.
        *   If $C$ is empty or $F > C_{fitness}$:
            *   Store $\theta$ in $C$.
            *   Save Snapshot to `universe_atlas.json`.
4.  **Mutation (Evolution):**
    *   Select random elites from the Archive.
    *   Mutate them (add Gaussian noise to $\theta$).
    *   Repeat Evaluation.

## 3. Deliverables
*   `scripts/explore_universe.py`: The implementation of the Explorer.
*   `output/universe_atlas.json`: The resulting catalog of universes.
