// src/cpp_core/src/harmonic_vacuum.cpp
#include "../include/sparse_engine.h"
#include <cstdint>
#include <functional>

namespace atheria {

HarmonicVacuum::HarmonicVacuum(int64_t d_state, torch::Device device)
    : d_state_(d_state), device_(device) {
}

torch::Tensor HarmonicVacuum::get_fluctuation(const Coord3D& coord, int64_t step_count) {
    // Generar semilla determinista a partir de coordenadas y tiempo
    // Usamos una función hash simple para garantizar consistencia
    std::hash<int64_t> hasher;
    size_t seed = hasher(coord.x) ^ (hasher(coord.y) << 1) ^ 
                  (hasher(coord.z) << 2) ^ (hasher(step_count) << 3);
    
    // Usar la semilla para generar ruido determinista
    // Establecer semilla global temporalmente
    // Nota: Esto afecta todas las operaciones aleatorias, pero es la forma más simple
    // TODO: Mejorar para usar generador específico si LibTorch lo permite
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_)
        .requires_grad(false);
    
    // Guardar semilla anterior
    auto prev_seed = torch::globalContext().defaultGenerator(device_).current_seed();
    
    // Establecer nueva semilla
    torch::manual_seed(static_cast<int64_t>(seed));
    
    // Generar ruido
    torch::Tensor noise = torch::randn({d_state_}, options) * 0.001f;
    
    // Restaurar semilla anterior
    torch::manual_seed(prev_seed);
    
    return noise;
}

} // namespace atheria

