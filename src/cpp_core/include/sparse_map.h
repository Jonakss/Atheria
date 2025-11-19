// src/cpp_core/include/sparse_map.h
#ifndef ATHERIA_SPARSE_MAP_H
#define ATHERIA_SPARSE_MAP_H

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstddef>

// Incluir LibTorch si está disponible
#ifdef TORCH_FOUND
#include <torch/torch.h>
#endif

namespace atheria {

/**
 * Estructura para coordenadas 3D
 */
struct Coord3D {
    int64_t x, y, z;
    
    Coord3D() : x(0), y(0), z(0) {}
    Coord3D(int64_t x, int64_t y, int64_t z) : x(x), y(y), z(z) {}
    
    bool operator==(const Coord3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * Hash function para Coord3D
 */
struct Coord3DHash {
    std::size_t operator()(const Coord3D& coord) const {
        // Hash combinado usando números primos
        return ((coord.x * 73856093) ^ (coord.y * 19349663) ^ (coord.z * 83492791));
    }
};

/**
 * SparseMap: Estructura de datos para mapeo disperso de alta performance.
 * 
 * Esta clase será el núcleo del motor disperso de Atheria 4.
 * Permite acceso rápido a elementos dispersos usando hash maps internos.
 * 
 * Versión con soporte para tensores de PyTorch (LibTorch).
 */
class SparseMap {
public:
    SparseMap();
    ~SparseMap();
    
    // Constructor de copia y asignación
    SparseMap(const SparseMap& other);
    SparseMap& operator=(const SparseMap& other);
    
    // Operaciones básicas con valores numéricos (compatibilidad hacia atrás)
    void insert(int64_t key, double value);
    bool contains(int64_t key) const;
    double get(int64_t key, double default_value = 0.0) const;
    void remove(int64_t key);
    void clear();
    
    // Operaciones con coordenadas 3D y tensores
#ifdef TORCH_FOUND
    void insert_tensor(const Coord3D& coord, const torch::Tensor& tensor);
    bool contains_coord(const Coord3D& coord) const;
    torch::Tensor get_tensor(const Coord3D& coord) const;
    void remove_coord(const Coord3D& coord);
#endif
    
    // Información del mapa
    size_t size() const;
    bool empty() const;
    
    // Iteración (para bindings Python)
    std::vector<int64_t> keys() const;
    std::vector<double> values() const;
    
#ifdef TORCH_FOUND
    std::vector<Coord3D> coord_keys() const;
#endif
    
private:
    // Almacenamiento para valores numéricos (compatibilidad hacia atrás)
    std::unordered_map<int64_t, double> data_;
    
#ifdef TORCH_FOUND
    // Almacenamiento para tensores
    std::unordered_map<Coord3D, torch::Tensor, Coord3DHash> tensor_data_;
#endif
};

} // namespace atheria

#endif // ATHERIA_SPARSE_MAP_H
