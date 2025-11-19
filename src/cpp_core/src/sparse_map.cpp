// src/cpp_core/src/sparse_map.cpp
#include "../include/sparse_map.h"
#include <algorithm>

namespace atheria {

SparseMap::SparseMap() = default;

SparseMap::~SparseMap() = default;

SparseMap::SparseMap(const SparseMap& other) 
    : data_(other.data_)
#ifdef TORCH_FOUND
    , tensor_data_(other.tensor_data_)
#endif
{}

SparseMap& SparseMap::operator=(const SparseMap& other) {
    if (this != &other) {
        data_ = other.data_;
#ifdef TORCH_FOUND
        tensor_data_ = other.tensor_data_;
#endif
    }
    return *this;
}

// Operaciones básicas con valores numéricos (compatibilidad hacia atrás)
void SparseMap::insert(int64_t key, double value) {
    data_[key] = value;
}

bool SparseMap::contains(int64_t key) const {
    return data_.find(key) != data_.end();
}

double SparseMap::get(int64_t key, double default_value) const {
    auto it = data_.find(key);
    return (it != data_.end()) ? it->second : default_value;
}

void SparseMap::remove(int64_t key) {
    data_.erase(key);
}

void SparseMap::clear() {
    data_.clear();
#ifdef TORCH_FOUND
    tensor_data_.clear();
#endif
}

size_t SparseMap::size() const {
    size_t total = data_.size();
#ifdef TORCH_FOUND
    total += tensor_data_.size();
#endif
    return total;
}

bool SparseMap::empty() const {
    bool is_empty = data_.empty();
#ifdef TORCH_FOUND
    is_empty = is_empty && tensor_data_.empty();
#endif
    return is_empty;
}

std::vector<int64_t> SparseMap::keys() const {
    std::vector<int64_t> result;
    result.reserve(data_.size());
    for (const auto& pair : data_) {
        result.push_back(pair.first);
    }
    return result;
}

std::vector<double> SparseMap::values() const {
    std::vector<double> result;
    result.reserve(data_.size());
    for (const auto& pair : data_) {
        result.push_back(pair.second);
    }
    return result;
}

#ifdef TORCH_FOUND
// Operaciones con coordenadas 3D y tensores
void SparseMap::insert_tensor(const Coord3D& coord, const torch::Tensor& tensor) {
    // torch::Tensor es un smart pointer, así que podemos copiarlo por valor de forma segura
    tensor_data_[coord] = tensor;
}

bool SparseMap::contains_coord(const Coord3D& coord) const {
    return tensor_data_.find(coord) != tensor_data_.end();
}

torch::Tensor SparseMap::get_tensor(const Coord3D& coord) const {
    auto it = tensor_data_.find(coord);
    if (it != tensor_data_.end()) {
        return it->second;  // torch::Tensor se copia de forma segura
    }
    // Retornar tensor vacío si no se encuentra
    return torch::Tensor();
}

void SparseMap::remove_coord(const Coord3D& coord) {
    tensor_data_.erase(coord);
}

std::vector<Coord3D> SparseMap::coord_keys() const {
    std::vector<Coord3D> result;
    result.reserve(tensor_data_.size());
    for (const auto& pair : tensor_data_) {
        result.push_back(pair.first);
    }
    return result;
}
#endif

} // namespace atheria
