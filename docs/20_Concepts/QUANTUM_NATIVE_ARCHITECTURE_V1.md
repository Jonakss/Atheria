# Quantum Native Architecture: The Q-U-Net

**Date:** 2024-05-22
**Status:** Conceptual Design / Experimental Phase
**Objective:** Transition Aetheria from a classical simulation (Complex Numbers in PyTorch) to a **Quantum Native** simulation where the state is represented by real qubits.

## 1. The Core Concept: Quantum Native State

In the classical "Law M", the state of a cell $(x, y)$ is a vector $\psi \in \mathbb{C}^d$.
In the **Quantum Native** architecture, the state is a **Density Matrix** $\rho$ or a **State Vector** $|\psi\rangle$ of a system of $n$ qubits.

*   **Classical:** `d_state` = 16 channels (float32).
*   **Quantum:** `d_state` = 4 Qubits (Hilbert Space dim = $2^4 = 16$).

### Advantages
1.  **Superposition:** The system can explore multiple "possible physics" simultaneously.
2.  **Entanglement:** Non-local correlations between cells can be modeled natively, mimicking the "AdS/CFT" holographic principle where boundary correlations encode bulk geometry.
3.  **True Randomness:** Instead of pseudo-random generators, we use quantum measurement collapse.

## 2. The Challenge: Scalability

Simulating a 128x128 grid where each cell has 4 qubits is impossible on current hardware.
*   Total Qubits: $128 \times 128 \times 4 \approx 65,000$ qubits.
*   Memory: $2^{65000}$ amplitudes. (Impossible).

### Solution: The Quantum Kernel (Sliding Window)

Instead of simulating the entire universe as one giant quantum circuit, we use a **Quantum Kernel**.
This is analogous to a CNN Kernel (Convolutional Neural Network).

*   **Window Size:** $3 \times 3$ cells.
*   **Input:** $3 \times 3 \times C$ classical features.
*   **Process:**
    1.  **Encode:** Embed classical features into a quantum state of $N$ qubits (Angle Embedding).
    2.  **Evolve:** Apply a parameterized Unitary $U(\theta)$ (The "Quantum Law").
    3.  **Entangle:** Use CNOT/CZ gates to mix information within the kernel.
    4.  **Measure:** Collapse the state to obtain new classical features.
*   **Output:** The center cell's new state.

This allows us to scan the "Quantum Law" across the classical grid, effectively creating a **Quantum Convolutional Layer**.

## 3. Architecture: The Q-U-Net (Hybrid U-Net)

We replace the bottleneck of our classical U-Net with a Quantum Circuit to maximize the "mixing" of information.

```mermaid
graph TD
    Input[Input Grid 128x128] --> Encoder[Classical Encoder (Conv2d)]
    Encoder --> |Downsample| Latent[Latent Space 8x8]
    Latent --> QLayer[Quantum Kernel Layer (VQC)]
    QLayer --> |Entanglement & Evolution| QLatent[Quantum Processed Latent]
    QLatent --> Decoder[Classical Decoder (ConvTranspose2d)]
    Decoder --> |Upsample| Output[Output Grid 128x128]
```

### Components
*   **Encoder:** Reduces the dimensionality of the "Bulk" space to a manageable "Boundary" size (Holographic Principle).
*   **Quantum Bottleneck (The Brain):**
    *   Input: $8 \times 8$ grid.
    *   Kernel: VQC interacting with neighboring cells.
    *   Function: Performs the complex non-linear dynamics that define the universe's evolution.
*   **Decoder:** Projects the quantum evolution back to the visualizable macroscopic universe.

## 4. Implementation Strategy (PennyLane + PyTorch)

We use **PennyLane** integrated with **PyTorch**.

*   `pennylane.qnode`: Defines the quantum circuit.
*   `pennylane.qnn.TorchLayer`: Converts the QNode into a standard `torch.nn.Module`.
*   **Gradients:** PennyLane handles backpropagation through the quantum circuit (using Parameter Shift Rule or Adjoint Differentiation), allowing us to **train** the laws of physics using classical gradient descent.

## 5. Next Steps

1.  **PoC:** Simulate a single "Quantum Cell" to prove probability conservation.
2.  **Integration:** Build the `QuantumKernel` module in `src/qca_engine_pennylane.py`.
3.  **Training:** Train the Q-U-Net to replicate known interesting patterns (Game of Life, Reaction-Diffusion) before letting it evolve autonomously.
