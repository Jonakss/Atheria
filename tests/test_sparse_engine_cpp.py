#!/usr/bin/env python3
"""
Script para probar el SparseQuantumEngineCpp y compararlo con la versión Python.

Permite probar las funcionalidades del v3 usando el núcleo C++.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import time
import atheria_core
from src.engines.sparse_engine import SparseQuantumEngine
from src.engines.sparse_engine_cpp import SparseQuantumEngineCpp

def test_basic_functionality():
    """Test básico de funcionalidad"""
    print("=" * 60)
    print("Test 1: Funcionalidad Básica")
    print("=" * 60)
    
    d_state = 4
    device = 'cpu'
    
    # Crear motor C++
    engine_cpp = SparseQuantumEngineCpp(None, d_state, device)
    
    # Crear estado de prueba
    state = torch.randn(d_state, device=device) * 0.1
    
    # Agregar partícula
    engine_cpp.add_particle((10, 20, 30), state)
    
    # Obtener estado
    retrieved = engine_cpp.get_state_at((10, 20, 30))
    
    print(f"Estado agregado: {state.shape}")
    print(f"Estado recuperado: {retrieved.shape if retrieved is not None else None}")
    print(f"Partículas almacenadas: {engine_cpp.get_matter_count()}")
    print()

def test_vacuum():
    """Test del vacío cuántico"""
    print("=" * 60)
    print("Test 2: Vacío Cuántico")
    print("=" * 60)
    
    d_state = 4
    device = 'cpu'
    
    engine_cpp = SparseQuantumEngineCpp(None, d_state, device)
    
    # Obtener estado del vacío en varias posiciones
    coords_list = [(0, 0, 0), (100, 200, 300), (1000, 2000, 3000)]
    
    print("Estados del vacío en diferentes coordenadas:")
    for coords in coords_list:
        vacuum_state = engine_cpp.get_state_at(coords)
        energy = torch.sum(vacuum_state.abs().pow(2)).item()
        print(f"  {coords}: energía = {energy:.6f}")
    print()

def test_step_simulation():
    """Test de simulación paso a paso"""
    print("=" * 60)
    print("Test 3: Simulación Paso a Paso")
    print("=" * 60)
    
    d_state = 4
    device = 'cpu'
    
    engine_cpp = SparseQuantumEngineCpp(None, d_state, device)
    
    # Agregar varias partículas
    particles = [
        ((10, 20, 30), torch.randn(d_state, device=device) * 0.5),
        ((50, 60, 70), torch.randn(d_state, device=device) * 0.5),
        ((100, 200, 300), torch.randn(d_state, device=device) * 0.5),
    ]
    
    for coords, state in particles:
        engine_cpp.add_particle(coords, state)
    
    print(f"Partículas iniciales: {engine_cpp.get_matter_count()}")
    
    # Ejecutar varios pasos
    for step in range(5):
        count = engine_cpp.step()
        print(f"Paso {step + 1}: {count} partículas activas")
    
    print()

def test_performance_comparison():
    """Comparación de rendimiento entre C++ y Python"""
    print("=" * 60)
    print("Test 4: Comparación de Rendimiento")
    print("=" * 60)
    
    d_state = 4
    device = 'cpu'
    num_particles = 1000
    
    # Test con C++
    engine_cpp = SparseQuantumEngineCpp(None, d_state, device)
    start = time.time()
    for i in range(num_particles):
        state = torch.randn(d_state, device=device) * 0.1
        engine_cpp.add_particle((i, i*2, i*3), state)
    cpp_insert_time = time.time() - start
    
    start = time.time()
    for i in range(num_particles):
        _ = engine_cpp.get_state_at((i, i*2, i*3))
    cpp_access_time = time.time() - start
    
    # Test con Python
    engine_py = SparseQuantumEngine(None, d_state, device)
    start = time.time()
    for i in range(num_particles):
        state = torch.randn(d_state, device=device) * 0.1
        engine_py.add_particle((i, i*2, i*3), state)
    py_insert_time = time.time() - start
    
    start = time.time()
    for i in range(num_particles):
        _ = engine_py.get_state_at((i, i*2, i*3))
    py_access_time = time.time() - start
    
    print(f"Insertar {num_particles} partículas:")
    print(f"  C++: {cpp_insert_time:.4f}s")
    print(f"  Python: {py_insert_time:.4f}s")
    if cpp_insert_time > 0:
        print(f"  Velocidad: {py_insert_time/cpp_insert_time:.2f}x {'más rápido' if py_insert_time < cpp_insert_time else 'más lento'}")
    
    print(f"\nAcceso a {num_particles} partículas:")
    print(f"  C++: {cpp_access_time:.4f}s")
    print(f"  Python: {py_access_time:.4f}s")
    if cpp_access_time > 0:
        print(f"  Velocidad: {py_access_time/cpp_access_time:.2f}x {'más rápido' if py_access_time < cpp_access_time else 'más lento'}")
    print()

def test_step_performance():
    """Comparación de rendimiento en step()"""
    print("=" * 60)
    print("Test 5: Rendimiento de step()")
    print("=" * 60)
    
    d_state = 4
    device = 'cpu'
    num_particles = 500
    num_steps = 10
    
    # Setup C++
    engine_cpp = SparseQuantumEngineCpp(None, d_state, device)
    for i in range(num_particles):
        state = torch.randn(d_state, device=device) * 0.5
        engine_cpp.add_particle((i*10, i*20, i*30), state)
    
    start = time.time()
    for _ in range(num_steps):
        engine_cpp.step()
    cpp_step_time = time.time() - start
    
    # Setup Python
    engine_py = SparseQuantumEngine(None, d_state, device)
    for i in range(num_particles):
        state = torch.randn(d_state, device=device) * 0.5
        engine_py.add_particle((i*10, i*20, i*30), state)
    
    start = time.time()
    for _ in range(num_steps):
        engine_py.step()
    py_step_time = time.time() - start
    
    print(f"Ejecutar {num_steps} pasos con {num_particles} partículas:")
    print(f"  C++: {cpp_step_time:.4f}s")
    print(f"  Python: {py_step_time:.4f}s")
    if cpp_step_time > 0:
        print(f"  Velocidad: {py_step_time/cpp_step_time:.2f}x {'más rápido' if py_step_time < cpp_step_time else 'más lento'}")
    print()

def main():
    """Función principal"""
    print("\n" + "=" * 60)
    print("🧪 Test de SparseQuantumEngineCpp")
    print("=" * 60)
    print()
    
    try:
        # Verificar que atheria_core está disponible
        if not atheria_core:
            print("⚠️  atheria_core no está disponible. Usando fallback a Python.")
        else:
            print("✅ atheria_core disponible")
        print()
        
        test_basic_functionality()
        test_vacuum()
        test_step_simulation()
        test_performance_comparison()
        test_step_performance()
        
        print("=" * 60)
        print("✅ Todos los tests completados!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

