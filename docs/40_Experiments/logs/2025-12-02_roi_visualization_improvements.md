# 2025-12-02 - Feature: ROI Visualization Improvements

## Context
As simulation grids grow larger (e.g., 512x512+), transferring the full state to the frontend becomes a bottleneck. We introduced a Region of Interest (ROI) system in the backend, but the frontend lacked proper controls and visualization for it. Users needed a way to toggle between the focused ROI view and the full grid view ("See All"), and a visual indication of when they were looking at a subset of the universe.

## Changes
### Backend
- **`src/pipelines/handlers/inference_handlers.py`**: Implemented `handle_set_roi_mode` command.
    - Allows toggling ROI on/off explicitly.
    - Broadcasts `roi_status_update` to all clients.
    - Forces an immediate frame update to reflect the change.

### Frontend
- **`src/context/WebSocketContext.tsx`**: 
    - Added `roiInfo` to the context state.
    - Implemented handler for `roi_status_update` messages.
- **`src/modules/Dashboard/components/VisualizationSection.tsx`**:
    - Added a "Control de Vista" section with a "See All" / "Focus View" toggle button.
    - Displays current ROI dimensions and reduction ratio.
- **`src/components/ui/PanZoomCanvas.tsx`**:
    - Implemented a gradient overlay effect that appears when ROI is active and `reduction_ratio > 1`.
    - Visual indicator "ROI ACTIVO" with dimensions.

### Fixes
- **`src/modules/PhaseSpaceViewer/PhaseSpaceViewer.tsx`**: Fixed a TypeScript build error related to `OrbitControls` types.

## Verification
- **Build**: `npm run build` passed successfully.
- **Manual**: Verified the toggle button sends the correct command and the overlay appears correctly when ROI is active.

## Next Steps
- Monitor performance with large grids to ensure the "See All" mode doesn't cause excessive lag (consider downsampling for full view in future).
