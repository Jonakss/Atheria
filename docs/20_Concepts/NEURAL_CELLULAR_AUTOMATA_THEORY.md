# Neural Cellular Automata (NCA): Theory & Differentiable Physics

## 1. Introduction: The Biological Metaphor
Neural Cellular Automata (NCA) represent a paradigm shift in how we model complex systems. Traditional physics simulations (like fluids or particles) rely on global equations (Navier-Stokes, Newton's laws) solved by a central "god-like" integrator. In contrast, **NCA models the world as a collective of independent agents (cells)**, much like biological tissues.

In biology, a single cell does not know the shape of the liver or the heart. It only knows its immediate chemical neighborhood and its own internal genetic code. Yet, these local interactions give rise to robust, self-repairing global structures.

**Aetheria** utilizes NCA to simulate a universe where "Physics" is not a fixed equation, but a **learned behavior** encoded in the weights of a neural network.

## 2. The Mathematical Model

### 2.1 The State Vector ($S_t$)
The universe is a grid of cells. Each cell $(i, j)$ holds a state vector $s_{i,j} \in \mathbb{R}^C$.
*   **Visible Channels (RGB/RGBA):** What we see.
*   **Hidden Channels ($h_1, ... h_n$):** Analogous to chemical concentrations, electric potentials, or morphogens. These channels carry the "memory" and "signaling" information required for complex behaviors (growth, regeneration).

### 2.2 Perception (The 'Sensors')
Before a cell can act, it must sense its environment. In NCA, this is modeled as a **Convolution**.
Unlike standard CNNs which reduce spatial dimensions, NCA perception preserves the grid size.
*   **Fixed Perception:** Often uses hardcoded filters (Sobel $K_x, K_y$, Laplacian) to detect gradients and edges. This mimics biological cells sensing chemical gradients.
*   **Learned Perception:** $3 \times 3$ Convolutional layers that learn *what* to look for in neighbors.

$$ P_{i,j} = \text{Concat}(s_{i,j}, \nabla_x s_{i,j}, \nabla_y s_{i,j}) $$

### 2.3 The Update Rule (The 'Genome')
The core "Physics" of the universe is a function $F_\theta$ (parameterized by a neural network) that maps the perception vector to a state update $\Delta s$.
Crucially, **$F_\theta$ is shared by every cell**. It is effectively the "DNA" of the universe.

$$ \Delta s_{i,j} = F_\theta(P_{i,j}) $$
$$ s_{t+1} = s_t + \Delta s_t \cdot \text{StochasticMask} $$

In Aetheria, this is typically implemented as a $1 \times 1$ Convolution (equivalent to a Dense layer applied per-pixel).

### 2.4 Stochasticity & Asynchrony
Real biological systems do not have a global clock tick. To prevent "perfect" but brittle structures (like Game of Life gliders that break with slight timing errors), NCA updates are **stochastic**.
*   **Dropout:** At each step, a random subset of cells is "frozen" (update mask = 0).
*   **Effect:** This forces the network to learn robust, self-correcting rules rather than relying on precise synchronization.

## 3. Differentiable Physics & Training

The power of NCA lies in **Differentiable Physics**. Because every step (Convolution, ReLU, Addition) is differentiable, we can unroll the simulation over time and backpropagate gradients through the entire timeline.

### 3.1 The Loop
1.  **Initial State ($S_0$):** A seed or random noise.
2.  **Unroll:** Run the NCA for $T$ steps to get $S_T$.
3.  **Loss Calculation:** Compare $S_T$ to a target (e.g., a specific pattern, or a state of high symmetry).
    $$ L = || S_T - \text{Target} ||^2 $$
4.  **Backpropagation Through Time (BPTT):** Calculate $\nabla_\theta L$. This tells us: *"How should I change the laws of physics (weights $\theta$) so that the universe ends up in the desired state?"*

### 3.2 Stability & Regeneration
To ensure the "Physics" are robust, we often employ:
*   **Pool Training:** Maintaining a pool of persistent states to learn long-term stability.
*   **Damage Training:** Intentionally zeroing out parts of the grid during training. The network learns to "regenerate" the missing information, effectively learning homeostasis.

## 4. Key References
*   *Mordvintsev et al., "Growing Neural Cellular Automata", Distill, 2020.*
*   *Randazzo et al., "Self-classifying MNIST Digits", Distill, 2020.*
*   *Gilpin, W. "Cellular automata as convolutional neural networks", Phys. Rev. E, 2019.*
