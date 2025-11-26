# Fix: FPS Calculation and Step Counting Logic

**Date:** 2025-11-26
**Author:** Antigravity (Google Deepmind)
**Status:** Completed
**Components:** Backend (Simulation Loop, Inference Handlers)

## Context
The user reported two main issues:
1.  **FPS Counter:** The FPS counter in the frontend was not updating (staying at 0 or static) when the live feed was enabled.
2.  **Step Counting:** When loading a trained model/checkpoint, the simulation step count would reset to 0, making it impossible to track the true progress of the simulation relative to its training history.

## Changes

### 1. FPS Calculation Fix (`src/pipelines/core/simulation_loop.py`)
- **Problem:** The FPS update logic was inside an `if not live_feed_enabled:` block, meaning it only updated when the live feed was OFF.
- **Fix:** Moved the FPS calculation and `g_state` update logic outside the conditional block. Now, `current_fps` is calculated and updated in `g_state` regardless of the live feed status.
- **Logic:**
    - Calculates `steps_per_second` based on actual execution time.
    - Updates `g_state['current_fps']` using a moving average window (10 samples) to smooth out fluctuations.

### 2. Step Counting Logic (`src/pipelines/handlers/inference_handlers.py` & `simulation_loop.py`)
- **Problem:** `handle_load_experiment` was hardcoding `g_state['simulation_step'] = 0` upon loading any model.
- **Fix:**
    - Modified `handle_load_experiment` to extract the step number from the filename:
        - For snapshots (`..._step_123.pt`): Extracts `123`.
        - For checkpoints (`..._ep100.pth`): Extracts episode `100` and multiplies by `QCA_STEPS_TRAINING` (default 100) to estimate the step.
    - Sets `g_state['simulation_step']` and `g_state['initial_step']` to this extracted value.
    - Updated `simulation_loop.py` to include `initial_step`, `session_steps` (current - initial), and `total_steps` (current) in the `simulation_info` payload sent to the frontend.

## Verification
- **Syntax Check:** `python3 -m py_compile ...` passed successfully.
- **Runtime Check:**
    - **FPS:** Should now update dynamically in the frontend when live feed is ON.
    - **Steps:** Loading a checkpoint should show the correct "Total Steps" (e.g., 10000 for ep100) and "Session Steps" starting from 0.

## Notes
- The local version in `setup.py` is static (`4.1.1`). Git tags are used for CI/CD versioning. This was clarified to the user.
