# Polar Architecture: The Phase-First Paradigm

**Date:** 2024-05-22
**Status:** Implementation Phase
**Objective:** Transition the core physics engine from Cartesian ($z = a + ib$) to Polar ($z = r e^{i\theta}$) representation to align with Quantum Hardware and U(1) Gauge Symmetry.

## 1. The Core Concept: Rotation is Fundamental

In Quantum Mechanics and Qiskit/PennyLane, the fundamental operation is the **Rotation** (e.g., $R_x(\theta)$, $R_z(\phi)$).
Our previous engine represented state as `[Real, Imag]`. While mathematically equivalent, it obscured the cyclic nature of the phase.

### The Shift
*   **Old (Cartesian):** State tensor shape `(B, 2, H, W)`. Channels 0=Real, 1=Imag.
*   **New (Polar):** State tensor shape `(B, 2, H, W)`. Channels 0=Magnitude ($r \in [0, 1]$), 1=Phase ($\theta \in [-\pi, \pi]$).

## 2. Physics & U(1) Symmetry

The laws of physics in Aetheria should respect **U(1) Symmetry**: A global rotation of the phase should not change the physical observables (like Energy or Magnitude distribution).
By treating $\theta$ explicitly, we can enforce this symmetry or study its breaking more easily.

### The Evolution Loop

While linear interference (superposition) happens in Cartesian space, non-linear interactions (The "Law M") often act on the Phase.

1.  **State:** $\psi = (r, \theta)$.
2.  **Spatial Mixing (Convolution):**
    *   Since $\sum r_i e^{i\theta_i} \neq (\sum r_i) e^{i (\sum \theta_i)}$, we **must** convert to Cartesian for the convolution step.
    *   $\psi_{cart} = \text{PolarToCart}(\psi)$.
    *   $\psi'_{cart} = \text{Conv2d}(\psi_{cart})$.
    *   $\psi' = \text{CartToPolar}(\psi'_{cart})$.
3.  **The "Law M" (Neural Update):**
    *   The neural network receives $(r', \theta')$.
    *   It predicts updates: $\Delta r, \Delta \theta$.
    *   **Crucial:** $\Delta \theta$ is a rotation.
    *   Update: $\theta_{new} = (\theta' + \Delta \theta) \pmod{2\pi}$.
    *   Update: $r_{new} = \text{Sigmoid}(r' + \Delta r)$.

## 3. Rotational U-Net

Standard ReLUs are bad for Phase angles because $f(2\pi) \neq f(0)$.
The **Rotational U-Net** solves this:
*   **Input Embedding:** Map $\theta \to [\sin(\theta), \cos(\theta)]$.
*   **Logic:** The network sees the "position on the circle", not the linear value.
*   **Output:** Predicts a scalar $\Delta \theta$.

## 4. Quantum Hardware Readiness

This architecture is "Quantum Native Ready":
*   **Angle Embedding:** We can feed $\theta$ directly into an $R_z(\theta)$ gate in a VQC.
*   **No Pre-processing:** No complex arctan calculations needed on the quantum chip or interface. The data is already in the language of rotations.

## 5. Visualization

Direct mapping:
*   **Hue:** $\theta$ (Phase).
*   **Brightness/Value:** $r$ (Magnitude).
*   **Saturation:** Constant or linked to stability.
This removes the costly `torch.atan2` from the visualization pipeline.
