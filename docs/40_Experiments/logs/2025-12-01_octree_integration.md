# 2025-12-01 - Octree Integration in Native Engine

## Summary
Integrated `OctreeIndex` into the C++ Native Engine (`atheria_core`) to enable efficient spatial queries and optimize simulation performance using Morton order (Z-curve) processing.

## Changes
- **C++ Core**:
    - Implemented `contains` and `query_box` in `OctreeIndex`.
    - Implemented `query_radius` in `Engine` using the Octree.
    - Optimized `step_native` to sort active particles by Morton code before processing, improving spatial locality.
- **Bindings**:
    - Exposed `query_radius` to Python.
- **Verification**:
    - Created `tests/test_octree_integration.py` verifying range queries and execution stability.

## Technical Details
- **Spatial Locality**: Sorting by Morton code ensures that particles close in 3D space are stored and processed contiguously (or close) in memory, reducing cache misses during neighbor lookups.
- **Range Queries**: `query_box` allows for efficient retrieval of particles within a bounding box, which is essential for "Moore neighborhood" operations and future optimizations like view frustum culling.
