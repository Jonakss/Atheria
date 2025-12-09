# 2025-12-09: Fix Training Crashes and 3D Controls Wiring

## Context
During debugging of the 3D Bulk visualization, we encountered two issues:
1.  **Backend Crashes:** The training loop in `QC_Trainer_v4` was crashing with `UnboundLocalError: psi` and `TypeError: ExperimentLogger.log_result`. These crashes likely prevented the simulation form updating, causing stale/flat data in the visualization.
2.  **Frontend Controls:** The "Point Size", "Threshold", and "Render Mode" controls in the Dashboard sidebar were not affecting the 3D visualization because their state was isolated in `VisualizationSection`.

## Changes

### 1. Backend: Training Stability
- **Fixed `UnboundLocalError: psi`:** In `src/trainers/qc_trainer_v4.py`, initialized `psi = psi_initial` before entering the simulation loop. This ensures `psi` is defined even if the loop logic is complex or skipped.
- **Fixed `loss_kwargs`:** Initialized `loss_kwargs = {}` in `qc_trainer_v4.py` to prevent `NameError`.
- **Fixed Logger Signature:** Updated `src/utils/experiment_logger.py` to accept the `snapshot_path` argument in `log_result`, matching the call site in the trainer.

### 2. Frontend: 3D Control Wiring
- **State Lifting:** Moved `pointSize`, `densityThreshold`, and `renderMode` state from `VisualizationSection` up to `DashboardLayout`.
- **Prop passing:** Passed these states down to:
    - `VisualizationSection` (as props to control the UI).
    - `HolographicViewer2` (as props to control the render).
    - `HolographicVolumeViewer` (as props).
- **Shader Update:** Updated `HolographicViewer2.tsx` to accepting a `uPointSize` uniform, allowing real-time adjustment of particle size.

## Verification
- **Training:** The backend crashes should be resolved, allowing the simulation loop to complete and send fresh data.
- **Visualization:** The "Bulk" view should now receive correct 3D data (fixed in previous step via `LatticeEngine` tensor dims) and respond to UI sliders.
