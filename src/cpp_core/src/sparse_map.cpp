// src/cpp_core/src/sparse_map.cpp
#include "../include/sparse_map.h"
#include <algorithm>

namespace atheria {

SparseMap::SparseMap() = default;

SparseMap::~SparseMap() = default;

SparseMap::SparseMap(const SparseMap& other) : data_(other.data_) {}

SparseMap& SparseMap::operator=(const SparseMap& other) {
    if (this != &other) {
        data_ = other.data_;
    }
    return *this;
}

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
}

size_t SparseMap::size() const {
    return data_.size();
}

bool SparseMap::empty() const {
    return data_.empty();
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

} // namespace atheria

