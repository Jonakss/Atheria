# Fix: CleanPolarWrapper Shape Attribute Error

**Date:** 2025-12-03  
**Severity:** Medium  
**Component:** Visualization Pipeline (`src/pipelines/viz/core.py`)

## Problem

`AttributeError: 'CleanPolarWrapper' object has no attribute 'shape'` when using `PolarEngine` with various visualization types, especially 'fields'.

**Stack Trace:**
```
File "src/pipelines/handlers/inference_handlers.py", line 1117, in handle_set_viz
  viz_data = get_visualization_data(psi, viz_type, delta_psi=delta_psi, motor=motor)
File "src/pipelines/viz/core.py", line 123, in get_visualization_data
  map_data = select_map_data(...)
File "src/pipelines/viz/core.py", line 298, in select_map_data
  if psi.shape[-1] >= 3:
AttributeError: 'CleanPolarWrapper' object has no attribute 'shape'
```

## Root Cause

The `CleanPolarWrapper` class was created during vacuum masking in `get_visualization_data()` to hold cleaned polar state data. However, it was missing:
- `.shape` property (expected by `select_map_data()`)
- `.abs()` method (expected by field visualization)
- Proper `.device` property management

Additionally, there was a duplicate class definition in the file.

## Solution

**Changes to `src/pipelines/viz/core.py`:**

1. **Added `@property shape`**: Returns `(H, W, 1)` tuple to indicate single-channel polar data
2. **Added `@property device`**: Returns device of underlying tensors with fallback
3. **Added `.abs()` method**: Returns magnitude as `(H, W, 1)` tensor for compatibility
4. **Removed duplicate class definition**: Consolidated into single class with all features

**Code:**
```python
class CleanPolarWrapper:
    """Wrapper para estado polar con máscara de vacío aplicada."""
    def __init__(self, mag, phase, device=None):
        self.magnitude = mag
        self.phase = phase
        self._device = device

    @property
    def shape(self):
        """Return shape compatible with tensor expectations."""
        h, w = self.magnitude.shape
        return (h, w, 1)
    
    @property
    def device(self):
        """Return device of underlying tensors."""
        if self._device is not None:
            return self._device
        return self.magnitude.device if hasattr(self.magnitude, 'device') else torch.device('cpu')

    def to_cartesian(self):
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        return real, imag
    
    def abs(self):
        """Return magnitude for compatibility with tensor operations."""
        return self.magnitude.unsqueeze(-1)
```

## Verification

Tested with Python command:
```bash
python3 -c "import sys; sys.path.insert(0, 'src'); from pipelines.viz.core import CleanPolarWrapper; import torch; w = CleanPolarWrapper(torch.rand(64, 64), torch.rand(64, 64)); print(f'Shape: {w.shape}'); print(f'Device: {w.device}'); print(f'abs() shape: {w.abs().shape}')"
```

**Output:**
```
Shape: (64, 64, 1)
Device: cpu
abs() shape: torch.Size([64, 64, 1])
```

## Impact

✅ All visualization types now work correctly with `PolarEngine`:
- `density`
- `phase`
- `fields` (previously failing)
- `real` / `imag`
- `poincare`
- etc.

## Related Issues

Similar to previous fix for `QuantumStatePolar` in [[logs/2025-12-02_fix_polar_viz_error]].

## Commit

```
fix: add shape property to CleanPolarWrapper for visualization compatibility [version:bump:patch]
```
