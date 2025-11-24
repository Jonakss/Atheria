# Live Feed Optimization & Fixes - Session Summary

**Date:** 2025-11-23
**Context:** This document summarizes critical fixes and performance optimizations implemented to resolve frontend-backend disconnection issues and progressive slowdowns during live simulation.

**⚠️ IMPORTANT FOR REFACTORING:** The following logic MUST be preserved or ported to the new `src/pipelines/core/simulation_loop.py` to ensure stability and performance.

## 1. Critical Performance Optimizations

To prevent network saturation and GPU memory leaks, we implemented a "compute-on-demand" strategy.

### A. Conditional Visualization Calculation (`pipeline_viz.py`)
Previously, ALL visualization data (Histograms, Poincaré, Flow, Phase Attractor) was calculated for every frame, regardless of what the user was viewing.
**Optimization:** We now pass `viz_type` to `get_visualization_data` and only calculate what is needed.

*   **Histograms:** Only calculated if `viz_type == 'histogram'`.
*   **Poincaré (PCA):** Only calculated if `viz_type` is `'poincare'` or `'poincare_3d'`.
*   **Phase Attractor:** Only calculated if `viz_type == 'phase_attractor'`.
*   **Flow Data:** Only calculated if `viz_type == 'flow'`.

### B. Dynamic Payload Construction (`pipeline_server.py`)
The WebSocket payload is now constructed dynamically to exclude unused data fields.
*   **Benefit:** Significantly reduces JSON size and serialization overhead.
*   **Implementation:** `frame_payload_raw` only includes keys like `complex_3d_data` or `phase_hsv_data` if the current `viz_type` requires them.

### C. GPU Memory Management
We observed progressive slowdowns when toggling the live feed.
**Fix:** Added explicit GPU cache clearing (`empty_cache_if_needed`) every **5 frames** within the simulation loop when live feed is active.

## 2. Live Feed & Turbo Mode Logic

### A. Broadcast Restoration
The simulation loop was missing the actual `broadcast()` call for simulation frames. This has been restored with proper `await` handling for the async `optimize_frame_payload` function.

### B. "Turbo Mode" Updates (Live Feed Disabled)
When `live_feed_enabled` is `False`:
*   **Behavior:** The simulation runs at maximum speed (0s sleep).
*   **Feedback:** We send a lightweight `simulation_step_update` message every **10 steps**.
*   **Frontend Handling:** The `WebSocketContext` processes this specific message type to update the step counter in the UI without rendering heavy graphics.

## 3. UX & Stability Improvements

*   **Manual Disconnect:** Added `disconnect()` to `WebSocketContext` and a button in the UI to allow resetting the connection if it gets stuck.
*   **Bug Fixes:**
    *   Fixed `AttributeError: 'list' object has no attribute 'tolist'` in Poincaré coordinates by checking types before serialization.
    *   Fixed `RuntimeWarning` by awaiting `optimize_frame_payload`.

## 4. Code References (Pre-Refactor)

*   **`src/pipeline_server.py`**: Contains the `simulation_loop` with the broadcast logic, turbo mode updates, and GPU cache clearing.
*   **`src/pipeline_viz.py`**: Contains the optimized `get_visualization_data` with conditional logic.
*   **`frontend/src/context/WebSocketContext.tsx`**: Handles `simulation_step_update` and manual disconnection.
