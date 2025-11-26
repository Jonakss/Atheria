#ifndef ATHERIA_CHUNK_MANAGER_H
#define ATHERIA_CHUNK_MANAGER_H

#include "sparse_map.h" // For Coord3D
#include <unordered_set>
#include <vector>
#include <functional>

namespace atheria {

struct ChunkCoord {
    int64_t x, y, z;

    bool operator==(const ChunkCoord& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct ChunkCoordHash {
    std::size_t operator()(const ChunkCoord& c) const {
        // Simple hash combination
        size_t h1 = std::hash<int64_t>{}(c.x);
        size_t h2 = std::hash<int64_t>{}(c.y);
        size_t h3 = std::hash<int64_t>{}(c.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class ChunkManager {
public:
    ChunkManager(int64_t chunk_size = 16);

    // Convert global coord to chunk coord
    ChunkCoord get_chunk_coord(const Coord3D& global_coord) const;

    // Get global origin of a chunk
    Coord3D get_chunk_origin(const ChunkCoord& chunk_coord) const;

    // Add active chunk (e.g. when particle is added)
    void activate_chunk(const ChunkCoord& chunk_coord);
    void activate_chunk_for_global_coord(const Coord3D& global_coord);

    // Get all active chunks
    const std::unordered_set<ChunkCoord, ChunkCoordHash>& get_active_chunks() const;

    // Clear state
    void clear();

    int64_t get_chunk_size() const { return chunk_size_; }

private:
    int64_t chunk_size_;
    std::unordered_set<ChunkCoord, ChunkCoordHash> active_chunks_;
};

} // namespace atheria

#endif // ATHERIA_CHUNK_MANAGER_H
