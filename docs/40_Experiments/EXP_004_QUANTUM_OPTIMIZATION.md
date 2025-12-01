# EXP-004: Quantum-Assisted Hyperparameter Optimization (QAHO)

**Date:** 2024-05-22
**Status:** In Progress
**Objective:** Leverage Quantum Variational Algorithms (VQE/QAOA) to optimize Aetheria's physics parameters ("Law M") for maximum emergent complexity.

## 1. Research: Formulating the Problem

The goal is to move from "running the simulation" to "optimizing the laws of physics". We treat the Aetheria simulation as a black-box function $f(\lambda)$ where $\lambda$ are the hyperparameters (e.g., `gamma_decay`) and the output is a "Complexity Score" (Entropy, Symmetry).

### A. QAOA (Quantum Approximate Optimization Algorithm)
**Use Case:** Discrete/Combinatorial Optimization.
**Application:** Finding stable *configurations* or *topologies*.
*   **Mapping:** If we discretize our hyperparameters into a grid (e.g., `gamma_decay` $\in \{0.0, 0.01, 0.1\}$), we can encode these choices into qubits.
    *   Let $q_0, q_1$ represent the binary encoding of `gamma_decay`.
*   **Hamiltonian Construction:**
    *   Constructing an Ising Hamiltonian $H_C$ directly from the Aetheria neural network weights is computationally infeasible due to the "Barren Plateau" problem and the sheer number of weights.
    *   **Strategy:** Instead, we use QAOA to optimize *meta-parameters* or *architecture choices* (e.g., Depth of U-Net, presence of Skip Connections) which can be mapped to a spin glass model.

### B. VQE (Variational Quantum Eigensolver)
**Use Case:** Continuous Optimization / Ground State Search.
**Application:** Tuning scalar hyperparameters (e.g., `gamma_decay`, `learning_rate`).
*   **Energy as Complexity:**
    *   Standard VQE minimizes Energy: $E = \langle \psi(\theta) | H | \psi(\theta) \rangle$.
    *   We define our Hamiltonian $H$ essentially as the "Lack of Complexity Operator".
    *   $H \equiv -1 \cdot \text{ComplexityMetric}(\text{Aetheria})$.
    *   Minimizing $E$ corresponds to Maximizing Complexity.
*   **The Challenge:** We cannot easily write the "Aetheria Operator" as a sum of Pauli strings ($\Sigma c_i P_i$).
*   **The Solution (Quantum-Assisted Optimization):**
    *   We use the Quantum Computer as a **Parametrized Distribution Generator**.
    *   The Quantum Circuit (Ansatz) $|\psi(\theta)\rangle$ generates a state.
    *   We measure this state to obtain a set of parameters $\lambda$.
    *   We run Aetheria classically with $\lambda$.
    *   We feed the result back to a classical optimizer (SPSA/COBYLA) which updates $\theta$ to find the "best region" in the Hilbert space that maps to high-complexity parameters.

## 2. Proposed Hybrid Workflow (The "Quantum Brain" Loop)

This architecture separates the "Dreamer" (Quantum) from the "Simulator" (Classical).

1.  **Quantum Suggestion (The Dream):**
    *   **Agent:** IBM Qiskit Runtime.
    *   **Action:** A Variational Circuit (Ansatz) parameterized by $\theta$ is executed.
    *   **Output:** We measure the qubits in the computational basis. The resulting bitstrings (or expectation values of Pauli-Z) are mapped to continuous hyperparameters (e.g., `gamma_decay` = $0.5 \cdot (1 + \langle Z_0 \rangle)$).

2.  **Classical Realization (The Simulation):**
    *   **Agent:** Aetheria Engine (GPU).
    *   **Action:** Run the simulation for $N=100$ steps using the suggested parameters.
    *   **Measurement:** Calculate the "Interesancia" (Reward) based on:
        *   $S$: Symmetry (visual coherence).
        *   $E$: Entropy (information density).
        *   $L$: Survival Rate (stability).
        *   **Reward Function:** $R = S \cdot E \cdot L$.

3.  **Feedback (The Learning):**
    *   **Agent:** Classical Optimizer (SPSA/COBYLA).
    *   **Action:** The Reward $R$ is treated as the negative energy (Cost = $-R$). The optimizer computes gradients (or approximations) and updates the circuit parameters $\theta$.
    *   **Goal:** The quantum state evolves to "concentrate" probability amplitude on the parameters that generate the most interesting universes.

## 3. Implementation Plan

We will implement a Proof of Concept (POC) script `scripts/quantum_optimize.py`.

*   **Libraries:** `qiskit`, `qiskit-algorithms`.
*   **Mocking:** Since running the full Aetheria engine in a loop is slow, the POC will use a **Mock Objective Function** that mimics the "Complexity Landscape" (e.g., a multi-modal function where we want to find the global maximum).
*   **Algorithm:** We will use a VQE-style loop where the Hamiltonian is implicit in the objective function.

### Next Steps for Production
1.  Integrate `scripts/quantum_optimize.py` with `src/engines/batch_inference_engine.py`.
2.  Define the `CostFunction` class to actually instantiate a `Universe` and run it.
