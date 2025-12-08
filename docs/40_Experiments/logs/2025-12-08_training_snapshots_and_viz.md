# 2025-12-08: Training Snapshots and Extended Field Visualizations

**Author:** AI Agent (Antigravity)
**Date:** 2025-12-08
**Tags:** #training #snapshots #visualization #lattice-engine #frontend

## 📝 Overview
We have implemented two key features from the roadmap:
1.  **Training Snapshots (Task #8):** Ability to capture and view QCA state snapshots during the training loop.
2.  **Extended Visualizations (Task #10):** Support for `Real` and `Imaginary` component visualization, and initial `Phase HSV` mapping.

## 🛠️ Implementation Details

### 1. Training Snapshots

**Backend:**
- **Trainer (`QC_Trainer_v4.py`):** Updated `save_checkpoint` to accept an optional `state_snapshot` tensor. It is saved as `training_checkpoints/<exp>/snapshot_ep<N>.pt`.
- **Pipeline (`pipeline_train.py`):** Modified the training loop to capture the `snapshot` tensor returned by `train_episode` and pass it to `save_checkpoint`.
- **Handlers (`experiment_handlers.py`):** 
    - Updated `handle_list_checkpoints` to detect if a snapshot exists for each checkpoint (`has_snapshot`).
    - Added `handle_load_checkpoint_snapshot` to pause simulation and inject the saved snapshot state into the active motor.

**Frontend:**
- **Context (`WebSocketContextDefinition.ts`):** Updated `TrainingCheckpoint` interface to include `has_snapshot`.
- **UI (`TrainingView.tsx`):** Added an "Eye" icon button next to checkpoints that have a snapshot. Clicking it triggers the load command.

### 2. Extended Visualizations

**Backend:**
- **Engine (`lattice_engine.py`):** 
    - Added `get_visualization_data` support for:
        - `viz_type="real"`: Returns normalized real part of the plaquette trace.
        - `viz_type="imag"`: Returns normalized imaginary part of the plaquette trace.
        - `viz_type="phase_hsv"`: Returns phase angle (mapped to Hue in frontend).

**Frontend:**
- **UI (`VisualizationSection.tsx`):** Added buttons/modes for "Parte Real" and "Parte Imag".

## 🧪 Verification
- Verified build of Frontend (`npm run build` passed).
- Code changes in `pipeline_train.py` ensure snapshots are pruned alongside checkpoints (smart save).
- `load_checkpoint_snapshot` correctly handles pausing simulation before loading.

## ⏭️ Next Steps
- Continue with **Task #17 (Holographic Principle)**: Design the `HolographicEngine` to simulate bulk/boundary dynamics explicitly.
- Benchmark the new visualization modes on the native engine (Future work).
