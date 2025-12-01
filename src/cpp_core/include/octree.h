#ifndef ATHERIA_OCTREE_H
#define ATHERIA_OCTREE_H

#include <vector>
#include <algorithm>
#include <cstdint>
#include "sparse_map.h" // For Coord3D

namespace atheria {

/**
 * Linear Octree Index using Morton Codes.
 * 
 * Stores a sorted list of Morton codes representing active particles.
 * Allows for efficient range queries and spatial lookups.
 */
class OctreeIndex {
public:
    OctreeIndex();
    ~OctreeIndex();

    /**
     * Insert a coordinate into the index.
     * Note: This does NOT sort immediately. Call build() after batch insertions.
     */
    void insert(const Coord3D& coord);

    /**
     * Sorts and deduplicates the codes. Must be called after insertions
     * and before queries to ensure correctness.
     */
    void build();

    /**
     * Clears the index.
     */
    void clear();

    /**
     * Returns the number of elements in the index.
     */
    // Check if a coordinate exists in the index
    bool contains(const Coord3D& coord) const;

    // Query all points within a bounding box (inclusive)
    std::vector<Coord3D> query_box(const Coord3D& min, const Coord3D& max) const;

    size_t size() const;

    /**
     * Checks if the index is empty.
     */
    bool empty() const;

    /**
     * Returns the raw sorted Morton codes (useful for debugging/visualization).
     */
    const std::vector<uint64_t>& get_codes() const;

    // Future: Range query methods
    // std::vector<Coord3D> query_radius(const Coord3D& center, double radius) const;

private:
    std::vector<uint64_t> codes_;
    bool is_built_;
};

} // namespace atheria

#endif // ATHERIA_OCTREE_H
