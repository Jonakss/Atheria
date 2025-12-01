# 2025-12-01 - Compute Backend Abstraction

**Status**: Completed
**Component**: Core Architecture

Implemented a modular "Compute Backend" system to abstract the execution hardware (Local CPU/GPU, Cloud Simulator, Real QPU).

## Changes
1.  **ComputeBackend Interface**: Defined abstract base class in `src/engines/compute_backend.py`.
2.  **Implementations**:
    - `LocalBackend`: Wraps PyTorch device (CPU/CUDA).
    - `MockQuantumBackend`: Simulates QPU connection for testing/UI.
3.  **ConnectionManager**: Service to manage backend registration and selection (`src/services/connection_manager.py`).
4.  **MotorFactory Refactor**: Updated `src/motor_factory.py` to use `ComputeBackend` and accept `BACKEND_TYPE` from config.
5.  **Trainer Integration**: Added `--backend_type` argument to `trainer.py`.

## Impact
- Decouples physics engine logic from specific hardware execution.
- Enables future integration of IonQ/IBM Quantum services without rewriting core engine logic.
- Allows "One-Click" switching between Local and Cloud backends (Lightning AI style).

## See also
- [[QUANTUM_COMPUTE_SERVICES]] - Research and integration strategy.
