# Python to Native Migration Guide

This document tracks the migration from the Python prototype engine to the high-performance Native C++ engine (`atheria_core`).

## Status

*   **Phase 1**: Initial C++ binding (Completed)
*   **Phase 2**: Optimization & Sparse Support (In Progress)
*   **Phase 3**: Full Parity & Deprecation of Python Engine (Planned)

## Troubleshooting Native Engine

If you encounter issues with the Native Engine, follow these steps.

### 1. Diagnosis
Run the diagnostic script to verify basic storage and retrieval functionality:

```bash
python3 test_native_state_recovery.py
```

*   **Success**: "DIAGNOSIS: Native engine is working correctly."
*   **Failure**: "DIAGNOSIS: Native engine has storage/retrieval issues." -> Check logs for "CRITICAL" errors.

### 2. Common Issues

#### Ghost Particles (Empty Visualization)
*   **Symptom**: Simulation runs (FPS > 0), `get_matter_count` > 0, but visualization is black.
*   **Cause**: `add_particle` in C++ might be failing to index particles correctly in the `matter_map_`, or `get_state_at` fails to retrieve them.
*   **Workaround**: The Python wrapper has been updated to verify particle addition immediately. If verification fails, it may attempt a fallback or warn you.

#### Timeouts on Play
*   **Symptom**: "WebSocket connection closed" when starting simulation on large grids.
*   **Cause**: Converting millions of particles from Sparse to Dense format takes time > 10s.
*   **Solution**: `inference_handlers.py` has been optimized to perform this asynchronously. Ensure you are using the latest version.

### 3. Force Python Engine
If the native engine is unstable, you can force the system to use the Python engine:

**Method A: Runtime (Switch)**
Use the "Switch Engine" button in the frontend (if available) or send command `switch_engine`.

**Method B: Configuration**
Edit `src/config.py`:
```python
USE_NATIVE_ENGINE = False
```

## Migration Checklist

- [x] Basic binding (`add_particle`, `get_state_at`)
- [x] Batch inference support
- [ ] Full parameter parity (Gamma, etc.)
- [x] Sparse-to-Dense conversion optimizations
- [ ] Save/Load state directly from C++
