# 2025-12-01 - Feature: Engine Selection for Training

**Author:** Agent
**Date:** 2025-12-01
**Tags:** #feature #training #backend #frontend

## Contexto
Previously, the training pipeline (`src/trainer.py`) did not allow selecting the physics engine type (`PYTHON`, `NATIVE`, `LATTICE`, `HARMONIC`), defaulting to `PYTHON` or relying on global config. This created a disparity with the inference mode, which supports engine switching.

## Cambios Implementados

### Backend
1.  **`src/trainer.py`**:
    - Added `--engine_type` argument to `argparse`.
    - Included `ENGINE_TYPE` in the experiment configuration dictionary (`exp_config`).
2.  **`src/server/server_handlers.py`**:
    - Updated `create_experiment_handler` to extract `ENGINE_TYPE` from the request arguments.
    - Passed `--engine_type` to the subprocess command launching `src.trainer`.

### Frontend
1.  **`frontend/src/components/experiments/TransferLearningWizard.tsx`**:
    - Updated `ExperimentConfig` interface to include `ENGINE_TYPE`.
    - Added a dropdown selector in the wizard UI to choose the engine type.
    - Passed the selected `ENGINE_TYPE` in the `experiment.create` WebSocket command.

## Verificación
- Verified that `src/trainer.py` accepts the new argument.
- Verified that the frontend sends the correct engine type.
- Fixed a syntax error in `TransferLearningWizard.tsx` (duplicate `</TableTd>`).

## Impacto
Users can now explicitly choose the simulation engine for their training experiments, enabling the use of optimized (Native) or specialized (Lattice, Harmonic) engines during the training phase.
