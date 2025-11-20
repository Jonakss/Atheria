// test_engine.cpp
// Test básico del motor nativo C++ (compilable con CMake o manualmente)
// Este archivo sirve como ejemplo de cómo usar el motor nativo desde C++

#include "../include/sparse_engine.h"
#include <iostream>
#include <cassert>

using namespace atheria;

int main() {
    std::cout << "🧪 Test del Motor Nativo C++" << std::endl;
    
    // Test 1: Crear motor
    std::cout << "\n1. Creando motor con d_state=8..." << std::endl;
    Engine engine(8, "cpu");
    std::cout << "✅ Motor creado exitosamente" << std::endl;
    
    // Test 2: Verificar contadores iniciales
    std::cout << "\n2. Verificando estado inicial..." << std::endl;
    assert(engine.get_matter_count() == 0);
    assert(engine.get_step_count() == 0);
    std::cout << "✅ Estado inicial correcto (0 partículas, 0 pasos)" << std::endl;
    
    // Test 3: Agregar partícula
    std::cout << "\n3. Agregando partícula..." << std::endl;
    Coord3D coord(10, 10, 0);
    auto options = torch::TensorOptions()
        .dtype(torch::kComplexFloat32)
        .device(torch::kCPU)
        .requires_grad(false);
    torch::Tensor state = torch::randn({8}, options) * 0.1f;
    engine.add_particle(coord, state);
    std::cout << "✅ Partícula agregada en (" << coord.x << ", " << coord.y << ", " << coord.z << ")" << std::endl;
    
    // Test 4: Verificar partícula agregada
    std::cout << "\n4. Verificando partícula..." << std::endl;
    assert(engine.get_matter_count() == 1);
    torch::Tensor retrieved_state = engine.get_state_at(coord);
    std::cout << "✅ Partícula encontrada. Estado shape: [" << retrieved_state.sizes() << "]" << std::endl;
    
    // Test 5: Ejecutar step sin modelo (solo conservar estado)
    std::cout << "\n5. Ejecutando step_native() sin modelo..." << std::endl;
    int64_t particles_after_step = engine.step_native();
    assert(engine.get_step_count() == 1);
    std::cout << "✅ Step ejecutado. Partículas después: " << particles_after_step << std::endl;
    
    // Test 6: Verificar HarmonicVacuum
    std::cout << "\n6. Probando HarmonicVacuum..." << std::endl;
    Coord3D vacuum_coord(5, 5, 0);
    torch::Tensor vacuum_state = engine.get_state_at(vacuum_coord);
    std::cout << "✅ Vacío generado. Estado shape: [" << vacuum_state.sizes() << "]" << std::endl;
    if (vacuum_state.is_complex()) {
        std::cout << "   Tipo: complejo ✓" << std::endl;
    } else {
        std::cout << "   Tipo: real" << std::endl;
    }
    
    std::cout << "\n✅ Todos los tests pasaron!" << std::endl;
    return 0;
}

