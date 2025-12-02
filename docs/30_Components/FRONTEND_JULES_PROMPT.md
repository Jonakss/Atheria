# Frontend Implementation Prompt: Quantum Features

**Context:**
We have implemented backend support for "Quantum Multiverse" (Batch Inference seeded by IonQ) and "Hybrid Simulation" (Runtime IonQ noise injection). We need to expose these controls in the UI.

**Objective:**
Update the frontend (`frontend/src/modules/ExperimentControl`, `LabSider`, etc.) to allow users to interact with these new quantum features.

## 1. Quantum Multiverse (Batch Mode)

**Location:** `NewExperimentModal` (when "Batch Mode" is selected).

**Requirements:**
-   **Toggle:** "Quantum Genesis (IonQ)"
    -   If ON: The "Random Seed" input should be disabled/hidden.
    -   **Description:** "Initialize all universes from a single quantum event (superposition collapse)."
    -   **API:** When creating the experiment, send `initial_mode='ionq'` and `batch_mode=True`.

**Visuals:**
-   Use a purple/quantum gradient border for this toggle to signify it's a premium/special feature.

## 2. Hybrid Simulation Controls

**Location:** `SimulationControls` (Right Sidebar or Bottom Bar).

**Requirements:**
-   **Toggle:** "Hybrid Injection" (Enable/Disable).
-   **Slider:** "Injection Interval" (10 - 100 steps). Default: 50.
-   **Slider:** "Quantum Noise Rate" (0.01 - 0.10). Default: 0.05.
-   **Action:** These params should be sent to the backend via WebSocket (`update_config` or similar) or passed during `step` calls.

## 3. Quantum Event Indicator

**Location:** `HolographicViewer` (Overlay).

**Requirements:**
-   When a "Quantum Injection" happens (every N steps), flash a subtle **Purple Ripple** effect over the viewport.
-   **Toast/Log:** Show "⚡ Quantum Injection" in the event log.

## Implementation Notes

-   **State:** Use `useExperimentStore` to track if `hybrid_mode` is active.
-   **WebSocket:** Listen for `quantum_injection` events from the backend to trigger the visual ripple.

---

**Tech Stack:** React, TypeScript, TailwindCSS, Zustand.
**Design System:** Atheria Dark Mode (Glassmorphism).
