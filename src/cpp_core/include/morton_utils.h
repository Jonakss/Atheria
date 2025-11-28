#ifndef ATHERIA_MORTON_UTILS_H
#define ATHERIA_MORTON_UTILS_H

#include <cstdint>
#include "sparse_map.h" // For Coord3D

namespace atheria {

/**
 * Utilities for 64-bit Morton Codes (Z-order curve).
 * 
 * We use 21 bits per dimension, which allows coordinates in range [0, 2,097,151].
 * This fits into a 63-bit integer (3 * 21 = 63).
 * 
 * Note: Coordinates are assumed to be non-negative for standard Morton coding.
 * If negative coordinates are needed, an offset must be applied before encoding.
 */

// Helper to expand bits (Magic numbers for bit interleaving)
// Expands a 21-bit integer into 64 bits by inserting 2 zeros after each bit.
inline uint64_t split_by_3(uint32_t a) {
    uint64_t x = a & 0x1fffff; // Ensure only 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8)  & 0x100f00f00f00f00f;
    x = (x | x << 4)  & 0x10c30c30c30c30c3;
    x = (x | x << 2)  & 0x1249249249249249;
    return x;
}

// Helper to compress bits (Reverse of split_by_3)
inline uint32_t compact_by_3(uint64_t x) {
    x &= 0x1249249249249249;
    x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
    x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
    x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
    x = (x ^ (x >> 16)) & 0x1f00000000ffff;
    x = (x ^ (x >> 32)) & 0x1fffff;
    return static_cast<uint32_t>(x);
}

// Encode 3D coordinates to a 64-bit Morton code
inline uint64_t coord_to_morton(int64_t x, int64_t y, int64_t z) {
    // Apply offset to handle negative coordinates if necessary.
    // For now, assuming raw coordinates or user handles offset.
    // Ideally, we should center the universe at (1<<20, 1<<20, 1<<20).
    // Let's assume input is already offset or non-negative for this low-level util.
    
    return split_by_3(static_cast<uint32_t>(x)) |
           (split_by_3(static_cast<uint32_t>(y)) << 1) |
           (split_by_3(static_cast<uint32_t>(z)) << 2);
}

// Decode a 64-bit Morton code to 3D coordinates
inline Coord3D morton_to_coord(uint64_t code) {
    uint32_t x = compact_by_3(code);
    uint32_t y = compact_by_3(code >> 1);
    uint32_t z = compact_by_3(code >> 2);
    return Coord3D(x, y, z);
}

} // namespace atheria

#endif // ATHERIA_MORTON_UTILS_H
