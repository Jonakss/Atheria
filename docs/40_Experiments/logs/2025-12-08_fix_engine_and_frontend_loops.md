# Fix Engine Crashes and Frontend Params

## Date: 2025-12-08
## Agent: Antigravity

## Issues Resolved
1. **LatticeEngine Crash**: `AttributeError: 'LatticeEngine' object has no attribute 'reset'`.
   - **Fix**: Added `reset()` method to `LatticeEngine` which calls `_initialize_links()`.

2. **HarmonicEngine Crash**: `AttributeError: 'SparseHarmonicEngine' object has no attribute 'steering'`.
   - **Fix**: Imported `IonQCollapse` and `QuantumSteering` and initialized them in `__init__`.

3. **Frontend Loop / Params Issue**: Repeated `set_click_out` calls with empty params.
   - **Fix**: Updated `VisualizationSection.tsx` to nest `enabled` and `chance` inside a `params` object when calling `sendCommand`.

## Affected Files
- `src/engines/lattice_engine.py`
- `src/engines/harmonic_engine.py`
- `frontend/src/modules/Dashboard/components/VisualizationSection.tsx`
- `frontend/src/context/WebSocketContext.tsx`
- `frontend/src/utils/timelineStorage.ts`
- `frontend/src/components/visualization/HolographicVolumeViewer.tsx`

4. **Frontend Errors & Storage Limit**:
   - **Issues**: `TypeError` on empty payload, `LocalStorage` quota exceeded loops, and `WebGL` shader error (invalid operation).
   - **Fix**: 
     - Added null check for `payload` in `WebSocketContext`.
     - Reduced `DEFAULT_MAX_FRAMES` to 25 and added loop protection in `timelineStorage`.
     - Renamed `size` -> `pSize` and added `precision mediump` in `HolographicVolumeViewer` shader.
     - Fixed syntax error (missing closing brace) in `HolographicViewer2.tsx` shader.
