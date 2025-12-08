# Optimization: Native Engine Performance Improvements

**Date:** 2025-12-08
**Tags:** #optimization #cpp #native-engine #openmp #pytorch

## Summary
Addressed critical performance bottlenecks in the Native Engine (C++) for both Sparse and Dense modes. Implemented in-place tensor operations, optimized batch construction, and resolved OpenMP deadlocks.

## Changes

### Sparse Engine (`sparse_engine.cpp`)
1.  **Vectorized Batch Input**: Replaced element-wise tensor filling with `torch::stack` and collected vectors, significantly reducing Python-C++ boundary overhead and internal loops.
2.  **Optimized Parallel Loop**:
    *   Replaced explicit `torch::complex` and other tensor allocations inside the OpenMP loop with in-place operations on reused tensors from `ThreadLocalTensorPool`.
    *   Used `torch::real(t)` and `torch::imag(t)` views for calculating updates without intermediate allocation.
    *   Used `torch::norm` instead of inefficient `abs().pow(2).sum()`.
3.  **Deadlock Fix**: Removed `torch::set_num_threads(1)` call inside OpenMP region which was causing deadlocks.

### Dense Engine (`dense_engine.cpp`)
1.  **Pre-allocated Buffers**: Added `input_buffer_` to avoid large allocations during `step()`.
2.  **In-place Operations**: 
    *   Used `state_.add_()` and `state_.div_()` instead of out-of-place operators.
    *   Used `torch::norm` for efficient normalization.

## Impact
- Reduced memory churn (allocations per step).
- Improved scalability with thread count (OpenMP).
- Expected significant speedup in `step_native` and `step`.
