// src/cpp_core/include/sparse_map.h
#ifndef ATHERIA_SPARSE_MAP_H
#define ATHERIA_SPARSE_MAP_H

#include <unordered_map>
#include <vector>
#include <cstdint>

namespace atheria {

/**
 * SparseMap: Estructura de datos para mapeo disperso de alta performance.
 * 
 * Esta clase será el núcleo del motor disperso de Atheria 4.
 * Permite acceso rápido a elementos dispersos usando hash maps internos.
 */
class SparseMap {
public:
    SparseMap();
    ~SparseMap();
    
    // Constructor de copia y asignación
    SparseMap(const SparseMap& other);
    SparseMap& operator=(const SparseMap& other);
    
    // Operaciones básicas
    void insert(int64_t key, double value);
    bool contains(int64_t key) const;
    double get(int64_t key, double default_value = 0.0) const;
    void remove(int64_t key);
    void clear();
    
    // Información del mapa
    size_t size() const;
    bool empty() const;
    
    // Iteración (para bindings Python)
    std::vector<int64_t> keys() const;
    std::vector<double> values() const;
    
private:
    std::unordered_map<int64_t, double> data_;
};

} // namespace atheria

#endif // ATHERIA_SPARSE_MAP_H

