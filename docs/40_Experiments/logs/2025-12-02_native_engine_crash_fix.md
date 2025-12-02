# 2025-12-02 - Critical Fix: Native Engine Crash & Quantum Tools Verification

## Context
The Native Engine (`atheria_core`) was crashing with `terminate called recursively` and `c10::Error` during simulation steps, specifically when running on CUDA with concurrent visualization access. This blocked the use of the high-performance C++ engine.

## Issue Analysis
1.  **Crash Symptoms**: Immediate crash upon starting simulation or shortly after, with `terminate called recursively`.
2.  **Root Cause Investigation**:
    -   Reproduction script confirmed the crash occurred on CUDA even with small grid sizes (64x64), ruling out OOM.
    -   Code analysis of `src/cpp_core/src/sparse_engine.cpp` revealed that the final batch processing block inside the OpenMP parallel region was **outside** the main `try-catch` block. Any exception thrown there (e.g., by PyTorch) would cause `std::terminate`.
    -   Code analysis of `src/cpp_core/src/harmonic_vacuum.cpp` showed `get_fluctuation` using a CPU generator (`torch::CPUGeneratorImpl`) with `torch::randn` on a CUDA device. Using a CPU generator for CUDA tensor creation inside parallel threads is risky and potentially undefined behavior in PyTorch, leading to exceptions.

## Resolution
1.  **Exception Safety**: Moved the `try-catch` block in `sparse_engine.cpp` to encompass the final batch processing logic. This ensures exceptions are caught and logged.
2.  **Thread Safety**: Modified `HarmonicVacuum::get_fluctuation` to generate noise tensors on the CPU first (using the thread-safe local CPU generator) and then move them to the target device (CUDA). This avoids any interaction between OpenMP threads and the CUDA generator state.

## Verification
-   **Reproduction**: A script `reproduce_crash.py` was created to mimic the crash conditions.
-   **Result**: After applying fixes and recompiling, the reproduction script ran successfully on CUDA for both 64x64 and 512x512 grid sizes.
-   **Quantum Tools**: Verified that `NativeEngineWrapper` correctly implements `apply_tool` and integrates with `IonQCollapse` and `QuantumSteering`.

## Related Changes
-   `src/cpp_core/src/sparse_engine.cpp`
-   `src/cpp_core/src/harmonic_vacuum.cpp`
-   `src/analysis/epoch_detector.py` (Fixed unrelated tensor shape mismatch)
