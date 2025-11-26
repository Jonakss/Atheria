# Debugging Grid, Canvas, and Versioning

**Date:** 2025-11-26
**Author:** Antigravity (AI Agent)
**Status:** Resolved

## Summary
Addressed three distinct issues related to simulation visualization, frontend rendering alignment, and development workflow versioning.

## Issues & Resolutions

### 1. Grid Size Defaulting to 256
**Problem:** The simulation always reset to a grid size of 256, ignoring the user-configured size (e.g., 128).
**Root Cause:** `server_handlers.py` was initializing `QuantumState` using `motor.grid_size` from the existing motor instance instead of the updated `GRID_SIZE_INFERENCE` from the global configuration.
**Fix:** Modified `handle_reset` in `src/server/server_handlers.py` to prioritize `global_cfg.GRID_SIZE_INFERENCE`.

### 2. Canvas Alignment (ROI)
**Problem:** When zooming in, the Region of Interest (ROI) appeared "desfasada" (misaligned), often jumping to (0,0) or not matching the grid lines.
**Root Cause:** The frontend `PanZoomCanvas.tsx` was rendering the ROI canvas layer at position (0,0) absolute, without accounting for the ROI's spatial offset (`x`, `y`) in the larger grid.
**Fix:** Updated `PanZoomCanvas.tsx` to apply `left` and `top` CSS styles to both the 2D `<canvas>` and the `ShaderCanvas` container based on `simData.roi_info`.

### 3. Versioning Inconsistency
**Problem:** `ath dev` was reinstalling version `4.1.1` despite `src/__version__.py` stating `4.2.0`.
**Root Cause:** `setup.py` had a hardcoded version string `4.1.1`.
**Fix:**
*   Updated `setup.py` to dynamically read the version from `src/__version__.py`.
*   Added `--skip-install` flag to `ath dev` in `src/cli.py` to allow skipping the redundant installation step during local development.

## Verification
*   **Versioning:** `ath version show` now correctly reports `4.2.0`. `ath dev --skip-install` skips the `pip install` step.
*   **Visuals:** ROI alignment and grid size persistence should be verified manually in the UI.
