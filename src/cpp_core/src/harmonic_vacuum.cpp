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
    
    // Generar en CPU primero para thread safety y evitar problemas con generadores CUDA en paralelo
    auto options_cpu = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU)
        .requires_grad(false);
    
    // Generar ruido complejo (similar a Python: complex_noise mode)
    // Python usa: noise = randn * strength; real = cos(noise); imag = sin(noise)
    const float complex_noise_strength = 0.1f;
    torch::Tensor noise = torch::randn({d_state_}, gen, options_cpu) * complex_noise_strength;
    
    // Convertir a complejo usando cos y sin
    torch::Tensor real = torch::cos(noise);
    torch::Tensor imag = torch::sin(noise);
    torch::Tensor complex_state_cpu = torch::complex(real, imag);
    
    // Mover al dispositivo destino (si es necesario)
    if (device_.type() != torch::kCPU) {
        return complex_state_cpu.to(device_);
    }
    
    return complex_state_cpu;
    

}

} // namespace atheria

