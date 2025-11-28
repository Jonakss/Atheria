#include "../include/octree.h"
#include "../include/morton_utils.h"

namespace atheria {

OctreeIndex::OctreeIndex() : is_built_(true) {}

OctreeIndex::~OctreeIndex() {}

void OctreeIndex::insert(const Coord3D& coord) {
    // Apply an offset to ensure coordinates are positive for Morton coding.
    // Assuming a "world center" at (0,0,0) maps to the middle of the Morton range.
    // Range is [0, 2^21 - 1]. Middle is 2^20 = 1048576.
    const int64_t OFFSET = 1048576; 
    
    // Clamp coordinates to valid range to prevent overflow
    int64_t x = std::max(int64_t(0), std::min(coord.x + OFFSET, int64_t(2097151)));
    int64_t y = std::max(int64_t(0), std::min(coord.y + OFFSET, int64_t(2097151)));
    int64_t z = std::max(int64_t(0), std::min(coord.z + OFFSET, int64_t(2097151)));

    uint64_t code = coord_to_morton(x, y, z);
    codes_.push_back(code);
    is_built_ = false;
}

void OctreeIndex::build() {
    if (is_built_) return;

    // Sort codes
    std::sort(codes_.begin(), codes_.end());

    // Remove duplicates (multiple particles in same coordinate shouldn't happen in SparseMap, 
    // but good for safety)
    auto last = std::unique(codes_.begin(), codes_.end());
    codes_.erase(last, codes_.end());

    is_built_ = true;
}

void OctreeIndex::clear() {
    codes_.clear();
    is_built_ = true;
}

size_t OctreeIndex::size() const {
    return codes_.size();
}

bool OctreeIndex::empty() const {
    return codes_.empty();
}

const std::vector<uint64_t>& OctreeIndex::get_codes() const {
    return codes_;
}

} // namespace atheria
