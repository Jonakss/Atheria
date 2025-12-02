# 2025-12-02 - Fix: Polar Engine Compatibility & Trainer Arguments

## Context
During training with the `PolarEngine`, two issues were encountered:
1. `AttributeError: 'QuantumStatePolar' object has no attribute 'clone'`: The trainer (`qc_trainer_v4.py`) attempts to clone the initial state for history tracking, but `QuantumStatePolar` lacked this method.
2. `WARNING:root:⚠️ Error cacheando checkpoint en Dragonfly: 'PolarEngine' object has no attribute 'operator'`: The trainer attempts to access `motor.operator` to save the model state dict, but `PolarEngine` didn't expose it.
3. `trainer.py: error: argument --backend_type: invalid choice: 'PYTHON'`: The training process failed to start because `PYTHON` (an engine type) was passed as `BACKEND_TYPE` to `trainer.py`.

## Changes
### Polar Engine (`src/engines/qca_engine_polar.py`)
- **Implemented `clone()`**: Added a method to `QuantumStatePolar` that creates a deep copy of the state (magnitude and phase tensors).
- **Added `operator` property**: Added a property to `PolarEngine` that aliases `self.model`, satisfying the interface expected by `QC_Trainer_v4`.

### Server Handlers (`src/server/server_handlers.py`)
- **Sanitized `BACKEND_TYPE`**: Modified `create_experiment_handler` to validate the `BACKEND_TYPE` argument. If the value is not in the valid list (e.g., if it's "PYTHON"), it defaults to "LOCAL". This prevents invalid arguments from crashing the trainer process.

## Verification
- **Manual**: Verified code changes. The `clone` method correctly copies tensors. The `operator` property correctly returns the model. The argument sanitization logic handles invalid inputs correctly.
