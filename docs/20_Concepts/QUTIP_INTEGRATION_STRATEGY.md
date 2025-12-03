# QuTiP Integration Strategy

## Overview

[QuTiP (Quantum Toolbox in Python)](https://qutip.org/) is a powerful open-source software for simulating the dynamics of open quantum systems. Unlike `qiskit` (circuit-based) or `pennylane` (variational quantum computing), QuTiP focuses on solving differential equations (Schrödinger, Lindblad) and manipulating density matrices directly.

For Project Aetheria, QuTiP serves a critical role in **verification** and **thermodynamic simulation**. While our [[PolarEngine]] and [[NativeEngine]] are optimized for speed and large 2D grids using effective field theories or tensor networks, QuTiP provides the "ground truth" for quantum mechanics on smaller scales, particularly for open systems (dissipation/noise).

## Strategic Fit

| Feature | [[PolarEngine]] (Current) | `PennyLaneEngine` | `QuTiPEngine` (Proposed) |
| :--- | :--- | :--- | :--- |
| **Representation** | Polar Tensors `(Mag, Phase)` | Quantum Circuits | Density Matrices `rho` |
| **Dynamics** | Unitary + Noise Injection | VQC / Unitary | Master Equation (`mesolve`) |
| **Primary Use** | Real-time Simulation, Visualization | Optimization (VQE), Hybrid AI | **Verification, Open Systems, Thermodynamics** |
| **Scalability** | High (Field Theory) | Medium (Simulator dependent) | Low (Exponential), High Fidelity |

## Key Use Cases

### 1. High-Fidelity Verification
Use QuTiP to verify that the [[PolarEngine]]'s effective updates approximate true unitary evolution for small $3 \times 3$ or $4 \times 4$ blocks.

### 2. Open Quantum Systems (Thermodynamics)
Aetheria explores the [[Harlow Limit]] and thermalization. QuTiP's `mesolve` allows us to simulate the Lindblad Master Equation:

$$ \dot{\rho} = -i[H, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho\} \right) $$

This allows for physically accurate modeling of:
*   Energy dissipation.
*   Decoherence rates.
*   Entropy production in the "Bulk".

### 3. Phase Space Visualization (Wigner Functions)
QuTiP has robust tools for generating Wigner functions (`qutip.wigner.wigner_3d`). We can use this to benchmark our own custom Wigner visualization pipeline.

## Implementation Plan

### Phase 1: Installation & Setup
Add `qutip` to `requirements.txt`.

```bash
pip install qutip
```

### Phase 2: The `QuTiPEngine`
Create `src/engines/qutip_engine.py` implementing the `PhysicsEngine` interface.

**Note on Terminology:** Aetheria's core engine typically refers to grid chunks as "Chunks" or uses hash maps for sparse storage. However, existing `PhysicsEngine` implementations (like `CartesianEngine`) utilize `grid_size` in their configuration and constructors. The `QuTiPEngine` wrapper will adapt this `grid_size` parameter to define the Hilbert space dimensions, keeping in mind QuTiP's exponential scaling makes it viable only for small "Chunks" or "Grids" (e.g., $3 \times 3$ or $4 \times 4$ sites).

```python
import qutip as qt
import numpy as np
from src.engines.qca_engine import PhysicsEngine

class QuTiPEngine(PhysicsEngine):
    def __init__(self, config):
        super().__init__(config)
        # 'grid_size' refers to the linear dimension of the lattice chunk (e.g., 3 or 4)
        self.dims = config.grid_size
        # Initialize Density Matrix
        self.state = qt.tensor([qt.basis(2, 0) for _ in range(self.dims**2)])
        self.hamiltonian = self._build_hamiltonian()

    def step(self, dt):
        # Use mesolve for one time step
        result = qt.mesolve(self.hamiltonian, self.state, [0, dt], c_ops=self.collapse_ops)
        self.state = result.states[-1]
        return self._to_tensor(self.state)
```

### Phase 3: Entanglement Entropy Benchmarks
Implement a benchmark script `scripts/verify_entropy.py` that compares:
1.  Aetheria's approximate local entropy.
2.  QuTiP's exact partial trace:
    ```python
    rho_sub = state.ptrace([0, 1]) # Trace out rest of universe
    entropy = qt.entropy_vn(rho_sub)
    ```

## Roadmap

1.  **Dependencies**: Update `requirements.txt` to include `qutip>=5.0.0`.
2.  **Prototype**: Build a standalone script `scripts/qutip_benchmark.py` to test performance limits on our hardware.
3.  **Integration**: Fully implement `QuTiPEngine` in the backend.
