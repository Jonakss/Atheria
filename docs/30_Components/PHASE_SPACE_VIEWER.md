# Phase Space Viewer

## Overview
The **Phase Space Viewer** is a visualization tool designed to analyze the topology of quantum states ($d_{state}$) in Aetheria. It provides two modes of operation:

1.  **Live Mode (PCA)**: A fast, linear dimensionality reduction using Principal Component Analysis (PCA). It runs in real-time (or near real-time) to show the evolution of the state's trajectory in phase space.
2.  **Analysis Mode (UMAP)**: A deeper, non-linear analysis using Uniform Manifold Approximation and Projection (UMAP). It is triggered on demand and reveals the detailed topological structure and clustering of the "matter" emerging in the simulation.

## Technical Implementation

### Backend (`src/pipelines/viz/phase_space.py`)
- **PCA**: Uses `sklearn.decomposition.PCA` to reduce the $d_{state}$ dimensions (e.g., 4 or 6) to 3 spatial dimensions (x, y, z).
- **UMAP**: Uses `umap-learn` for non-linear reduction. This is computationally intensive and runs in a separate thread to avoid blocking the simulation loop.
- **Clustering**: Applies K-Means clustering to color-code points based on their structural similarity.
- **Subsampling**: To maintain performance, only a subset of pixels (e.g., 10%) is analyzed, using a strided sampling approach.

### Frontend (`PhaseSpaceViewer.tsx`)
- **Technology**: Built with React Three Fiber (Three.js for React).
- **Visualization**: Renders the state as a 3D point cloud.
- **Interaction**: Supports OrbitControls for rotation, zoom, and panning.
- **Integration**: Accessible via the "Visualization" drawer in the Dashboard, selecting "Phase Space" as the visualization type.

## Usage
1.  Open the **Visualization** drawer (right sidebar).
2.  Select **Phase Space** from the visualization type dropdown.
3.  The view will default to PCA (if implemented for live feed) or wait for analysis.
4.  Click the **Deep Analyze (UMAP)** button to trigger a high-fidelity topological analysis of the current frame. The simulation will pause during this process.

## Dependencies
- **Backend**: `scikit-learn`, `umap-learn`
- **Frontend**: `@react-three/fiber`, `@react-three/drei`, `three`
