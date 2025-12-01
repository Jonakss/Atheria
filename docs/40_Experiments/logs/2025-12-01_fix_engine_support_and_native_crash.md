# Fix Engine Support and Native Crash Investigation

## Summary
Addressed issues with `LATTICE` and `HARMONIC` engine support in `motor_factory.py` and investigated a crash in the Native Engine (`atheria_core`).

## Changes
1.  **`src/motor_factory.py`**:
    *   Imported `LatticeEngine` and `SparseHarmonicEngine`.
    *   Added logic to instantiate `LatticeEngine` (no model required) and `SparseHarmonicEngine` (model required).
    *   Fixed `AttributeError: 'NoneType' object has no attribute 'eval'` by ensuring `LATTICE` engine doesn't try to use a `None` model in `Aetheria_Motor` fallback.

2.  **`src/engines/qca_engine_polar.py`**:
    *   Added `@property shape` to `QuantumStatePolar` to fix `AttributeError: 'QuantumStatePolar' object has no attribute 'shape'`.

## Native Engine Crash Investigation
*   **Symptom**: `terminate called recursively` and `Aborted (core dumped)` when running `NativeEngineWrapper`.
*   **Reproduction**: Created `debug_native.py` which reproduced the crash reliably during `motor.evolve_internal_state()`.
*   **Analysis**:
    *   The crash occurs when the C++ engine invokes the JIT-compiled PyTorch model.
    *   `verify_jit.py` confirmed the JIT model expects **Real Concatenated** input `[1, 2*d_state, H, W]`.
    *   Hypothesized that C++ engine passes **Complex** input `[1, d_state, H, W]` or `[1, H, W, d_state]`.
    *   Attempted to fix by wrapping the model in `NativeEngineWrapper.export_model_to_jit` to handle Complex <-> Real conversion and layout permutation.
    *   Tried both Channels First `[B, C, H, W]` and Channels Last `[B, H, W, C]` wrappers.
    *   **Result**: Both attempts failed with the same crash. The exact interface expected by `atheria_core` remains unclear without access to C++ source or debugging symbols.
*   **Status**: Native Engine crash remains unresolved. Recommended using Python engine (`PYTHON`, `POLAR`, `LATTICE`, `HARMONIC`) until C++ engine can be debugged.

## Verification
*   Verified `motor_factory.py` correctly creates `LATTICE` and `HARMONIC` engines using `verify_engines.py`.
