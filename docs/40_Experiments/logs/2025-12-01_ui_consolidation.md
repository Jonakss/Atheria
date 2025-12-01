
## 2025-12-01: UI Consolidation and Holographic Viewer Improvements

### UI Consolidation
- **Right Drawer**: Consolidated `VisualizationPanel` and `PhysicsInspector` into a single unified `RightDrawer` component.
- **VSCode-style**: Implemented collapsible drawer with icon-only mode and tabbed navigation (Visualization / Physics).
- **Cleanup**: Removed duplicate visualization controls from `LabSider` (left panel).

### Holographic Viewer Improvements
- **Controls**: Added new controls for 3D visualization:
  - **Render Mode**: Points, Wireframe, Mesh.
  - **Point Size**: Adjustable slider (0.5x - 5.0x).
  - **Threshold**: Adjustable density threshold (0.001 - 0.5).
- **Visualization**:
  - **Threshold**: Lowered default threshold to `0.01` to reveal more quantum structure.
  - **Color**: Adjusted shader colors to avoid "blanqueado" (bleached) effect, improving contrast and structure visibility.

### Engine Documentation
- **Compatibility Matrix**: Updated `ENGINE_COMPATIBILITY_MATRIX.md` to clarify Python-only vs C++ engine status.
- **Roadmap**: Added C++ implementation roadmap for Harmonic, Lattice, and Polar engines.
