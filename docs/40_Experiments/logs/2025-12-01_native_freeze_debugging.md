# Native Engine Freeze Debugging Attempt

**Date**: 2025-12-01  
**Objective**: Resolve Native Engine hang during warmup to enable Python vs C++ benchmark comparison

## Problem

Native Engine blocks during warmup in benchmark tests, preventing performance comparison with Python engine (baseline: ~60 FPS on CPU, 128x128 grid).

## Investigation

### Code Analysis
- Reviewed `native_engine_wrapper.py`: Uses `threading.RLock` for thread safety
- Reviewed `sparse_engine.cpp`: Uses OpenMP parallel regions with `torch::set_num_threads(1)` inside
- Reviewed benchmark script: Synchronous warmup loop with no yields
- **Suspected cause**: Python GIL contention with OpenMP threads during long C++ computations

### Initial Hypothesis
GIL (Global Interpreter Lock) was not being released during C++ `step_native()` execution, causing:
- Main Python thread blocked waiting for OpenMP workers
- OpenMP threads unable to acquire GIL for LibTorch operations
- Deadlock or severe contention

## Implementation

### 1. C++ Bindings - GIL Release Guard

**File**: `src/cpp_core/src/bindings.cpp` (line 150)

```cpp
.def("step_native", &Engine::step_native,
     py::call_guard<py::gil_scoped_release>(),  // Release GIL during C++ computation
     "Ejecuta un paso completo de simulación en C++ (todo el trabajo pesado)",
     "Retorna el número de partículas activas")
```

**Purpose**: Allow OpenMP threads to execute freely without Python GIL contention

### 2. Benchmark Script - Periodic Yields

**File**: `scripts/benchmark_comparison.py` (lines 63-70)

```python
# Warmup
print(f"Warming up ({warmup} steps)...")
for i in range(warmup):
    engine.evolve_internal_state(step=i)
    # Add periodic yield to prevent blocking and allow signal handlers
    if (i+1) % 10 == 0:
        time.sleep(0.001)  # 1ms yield
        print(f"  Warmup: {i+1}/{warmup}")
```

**Purpose**: Allow signal handlers to run and provide progress feedback

### 3. Module Recompilation

```bash
rm -rf build/ ath_venv/lib/python3.10/site-packages/atheria_core.*
python3 setup.py build_ext --inplace
```

**Result**: ✅ Successful compilation with no errors

## Testing

### Test 1: Full Benchmark
```bash
timeout 120 python3 -u scripts/benchmark_comparison.py
```
- **Result**: ❌ Hung without output for 3+ minutes
- **Exit**: Terminated manually

### Test 2: Quick Test (5 steps only)
```bash
timeout 30 python3 -u scripts/test_native_quick.py
```
- **Result**: ❌ Also hung without any output
- **Observation**: Even minimal test fails to produce console output

## Conclusion

**GIL release did NOT resolve the freeze.**

### Updated Analysis

The hang likely occurs **before** `step_native()` execution, during:
1. **JIT Model Export** (`export_model_to_jit`) - happens in offload thread
2. **Model Loading** (`engine.load_model()`) - LibTorch loading TorchScript
3. **Initial State Transfer** (`_initialize_native_state_from_dense`) - converting 128x128 grid to sparse particles

### Evidence
- Even 5-step test hangs (not related to warmup length)
- No console output produced at all (hang before first print statement)
- Previous tests showed `step_native()` working in isolation
- Grid size (128x128 = 16,384 cells) may cause initialization timeout

## Recommended Next Steps

1. **Add debug logging** to pinpoint exact hang location:
   ```python
   logging.info("DEBUG: Starting NativeEngineWrapper init...")
   logging.info("DEBUG: Creating native engine...")
   logging.info("DEBUG: Loading model...")
   ```

2. **Test with smaller grid** (32x32 or 64x64) to rule out scale issue

3. **Test C++ directly** without Python wrapper:
   ```python
   import atheria_core
   engine = atheria_core.Engine(d_state=64, device="cpu", grid_size=32)
   ```

4. **Consider deferring benchmark** until initialization issue is resolved

## Files Modified

- `src/cpp_core/src/bindings.cpp` - Added GIL release guard
- `scripts/benchmark_comparison.py` - Added periodic yields
- `scripts/test_native_quick.py` - Created quick test script

## Commits

```
dbf2ce6 - fix: add GIL release to step_native and yields to benchmark (issue persists) [version:bump:patch]
71926c9 - docs: actualizar estado del proyecto - Fase 2 benchmarking, Fase 3 completada, Compute Backend 100% [version:bump:patch]
```

## References

- [[2025-12-01_benchmarking]]: Original benchmark results (Python ~60 FPS, Native blocked)
- [[Native_Engine_Core]]: Native Engine architecture documentation
- [[PHASE_STATUS_REPORT]]: Current project status (Fase 2 at 85%)
