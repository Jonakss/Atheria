#include "../include/chunk_manager.h"

namespace atheria {

ChunkManager::ChunkManager(int64_t chunk_size) : chunk_size_(chunk_size) {}

ChunkCoord ChunkManager::get_chunk_coord(const Coord3D& global_coord) const {
    // Integer division with floor behavior for negative numbers
    auto floor_div = [](int64_t a, int64_t b) {
        return (a >= 0) ? (a / b) : ((a - b + 1) / b);
    };
    return {
        floor_div(global_coord.x, chunk_size_),
        floor_div(global_coord.y, chunk_size_),
        floor_div(global_coord.z, chunk_size_)
    };
}

Coord3D ChunkManager::get_chunk_origin(const ChunkCoord& chunk_coord) const {
    return Coord3D(
        chunk_coord.x * chunk_size_,
        chunk_coord.y * chunk_size_,
        chunk_coord.z * chunk_size_
    );
}

void ChunkManager::activate_chunk(const ChunkCoord& chunk_coord) {
    active_chunks_.insert(chunk_coord);
}

void ChunkManager::activate_chunk_for_global_coord(const Coord3D& global_coord) {
    activate_chunk(get_chunk_coord(global_coord));
}

const std::unordered_set<ChunkCoord, ChunkCoordHash>& ChunkManager::get_active_chunks() const {
    return active_chunks_;
}

void ChunkManager::clear() {
    active_chunks_.clear();
}

} // namespace atheria
