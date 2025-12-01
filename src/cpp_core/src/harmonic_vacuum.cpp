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
    // Crear generador local para evitar modificar estado global (thread-safe)
    auto gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_)
        .requires_grad(false);
    
    // Generar ruido complejo (similar a Python: complex_noise mode)
    // Python usa: noise = randn * strength; real = cos(noise); imag = sin(noise)
    const float complex_noise_strength = 0.1f;
    torch::Tensor noise = torch::randn({d_state_}, gen, options) * complex_noise_strength;
    
    // Convertir a complejo usando cos y sin (como en Python)
    torch::Tensor real = torch::cos(noise);
    torch::Tensor imag = torch::sin(noise);
    torch::Tensor complex_state = torch::complex(real, imag);
    
    return complex_state;
    

}

} // namespace atheria

