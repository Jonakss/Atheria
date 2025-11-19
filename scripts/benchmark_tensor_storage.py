#!/usr/bin/env python3
"""
Benchmark comparativo: Almacenamiento de tensores Python dict vs C++ SparseMap.

Este script compara el rendimiento entre:
1. Almacenamiento tradicional: Python dict {(x,y,z): tensor}
2. Almacenamiento nativo C++: SparseMap con Coord3D y torch::Tensor

Métricas comparadas:
- Tiempo de inserción
- Tiempo de acceso/recuperación
- Tiempo de iteración
- Uso de memoria (estimado)
"""
import sys
from pathlib import Path
import time
import torch
import numpy as np

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import atheria_core

def benchmark_python_dict(num_particles, tensor_size, d_state=4):
    """Benchmark usando diccionario Python tradicional"""
    print(f"\n📊 Benchmark: Python Dict (control)")
    print(f"   Partículas: {num_particles}, Tensor size: {tensor_size}")
    
    # Crear tensores
    tensors = []
    coords = []
    for i in range(num_particles):
        coords.append((i*10, i*20, i*30))
        tensors.append(torch.randn(d_state, dtype=torch.float32))
    
    # Test 1: Inserción
    storage = {}
    start = time.time()
    for coord, tensor in zip(coords, tensors):
        storage[coord] = tensor
    insert_time = time.time() - start
    
    # Test 2: Acceso aleatorio
    start = time.time()
    for i in range(0, num_particles, max(1, num_particles // 1000)):
        _ = storage[coords[i]]
    access_time = time.time() - start
    access_count = num_particles // max(1, num_particles // 1000)
    
    # Test 3: Verificación de existencia
    start = time.time()
    for i in range(0, num_particles, max(1, num_particles // 1000)):
        _ = coords[i] in storage
    contains_time = time.time() - start
    
    # Test 4: Iteración sobre todas las claves
    start = time.time()
    keys_list = list(storage.keys())
    keys_time = time.time() - start
    
    # Test 5: Eliminación
    start = time.time()
    for i in range(0, num_particles, max(1, num_particles // 10)):
        if coords[i] in storage:
            del storage[coords[i]]
    remove_time = time.time() - start
    remove_count = num_particles // max(1, num_particles // 10)
    
    return {
        'insert_time': insert_time,
        'access_time': access_time,
        'access_count': access_count,
        'contains_time': contains_time,
        'keys_time': keys_time,
        'remove_time': remove_time,
        'remove_count': remove_count,
        'size': len(storage)
    }

def benchmark_cpp_sparsemap(num_particles, tensor_size, d_state=4):
    """Benchmark usando SparseMap C++ con tensores nativos"""
    if not atheria_core.has_torch_support():
        print("❌ LibTorch no disponible, saltando benchmark C++")
        return None
    
    print(f"\n📊 Benchmark: C++ SparseMap (tensores nativos)")
    print(f"   Partículas: {num_particles}, Tensor size: {tensor_size}")
    
    # Crear tensores y coordenadas
    tensors = []
    coords = []
    for i in range(num_particles):
        coords.append(atheria_core.Coord3D(i*10, i*20, i*30))
        tensors.append(torch.randn(d_state, dtype=torch.float32))
    
    # Test 1: Inserción
    storage = atheria_core.SparseMap()
    start = time.time()
    for coord, tensor in zip(coords, tensors):
        storage.insert_tensor(coord, tensor)
    insert_time = time.time() - start
    
    # Test 2: Acceso aleatorio
    start = time.time()
    for i in range(0, num_particles, max(1, num_particles // 1000)):
        _ = storage.get_tensor(coords[i])
    access_time = time.time() - start
    access_count = num_particles // max(1, num_particles // 1000)
    
    # Test 3: Verificación de existencia
    start = time.time()
    for i in range(0, num_particles, max(1, num_particles // 1000)):
        _ = storage.contains_coord(coords[i])
    contains_time = time.time() - start
    
    # Test 4: Iteración sobre todas las claves
    start = time.time()
    keys_list = storage.coord_keys()
    keys_time = time.time() - start
    
    # Test 5: Eliminación
    start = time.time()
    for i in range(0, num_particles, max(1, num_particles // 10)):
        if storage.contains_coord(coords[i]):
            storage.remove_coord(coords[i])
    remove_time = time.time() - start
    remove_count = num_particles // max(1, num_particles // 10)
    
    return {
        'insert_time': insert_time,
        'access_time': access_time,
        'access_count': access_count,
        'contains_time': contains_time,
        'keys_time': keys_time,
        'remove_time': remove_time,
        'remove_count': remove_count,
        'size': storage.size()
    }

def benchmark_sparse_engine_operations(num_particles, steps=10, d_state=4):
    """Benchmark de operaciones típicas del motor disperso"""
    print(f"\n📊 Benchmark: Operaciones de Motor Disperso")
    print(f"   Partículas: {num_particles}, Pasos: {steps}")
    
    # Python Dict
    from src.engines.sparse_engine import SparseQuantumEngine
    
    engine_py = SparseQuantumEngine(None, d_state, 'cpu')
    for i in range(num_particles):
        state = torch.randn(d_state, dtype=torch.float32) * 0.1
        engine_py.add_particle((i*10, i*20, i*30), state)
    
    start = time.time()
    for _ in range(steps):
        engine_py.step()
    py_step_time = time.time() - start
    
    # C++ SparseMap (versión actual con diccionario auxiliar)
    from src.engines.sparse_engine_cpp import SparseQuantumEngineCpp
    
    engine_cpp = SparseQuantumEngineCpp(None, d_state, 'cpu')
    for i in range(num_particles):
        state = torch.randn(d_state, dtype=torch.float32) * 0.1
        engine_cpp.add_particle((i*10, i*20, i*30), state)
    
    start = time.time()
    for _ in range(steps):
        engine_cpp.step()
    cpp_step_time = time.time() - start
    
    return {
        'python_step_time': py_step_time,
        'cpp_step_time': cpp_step_time,
    }

def print_comparison(py_results, cpp_results):
    """Imprime comparación de resultados"""
    if cpp_results is None:
        print("\n⚠️  No hay resultados C++ para comparar")
        return
    
    print("\n" + "=" * 70)
    print("📈 COMPARACIÓN DE RENDIMIENTO")
    print("=" * 70)
    
    # Inserción
    print(f"\n1. INSERCIÓN:")
    print(f"   Python Dict:    {py_results['insert_time']:.6f}s")
    print(f"   C++ SparseMap:  {cpp_results['insert_time']:.6f}s")
    if cpp_results['insert_time'] > 0:
        speedup = py_results['insert_time'] / cpp_results['insert_time']
        print(f"   ⚡ Velocidad: {speedup:.2f}x {'más rápido' if speedup > 1 else 'más lento'} (C++)")
    
    # Acceso
    py_access_per_item = py_results['access_time'] / py_results['access_count']
    cpp_access_per_item = cpp_results['access_time'] / cpp_results['access_count']
    print(f"\n2. ACCESO ALEATORIO ({py_results['access_count']} accesos):")
    print(f"   Python Dict:    {py_results['access_time']:.6f}s ({py_access_per_item*1e6:.2f} μs/item)")
    print(f"   C++ SparseMap:  {cpp_results['access_time']:.6f}s ({cpp_access_per_item*1e6:.2f} μs/item)")
    if cpp_access_per_item > 0:
        speedup = py_access_per_item / cpp_access_per_item
        print(f"   ⚡ Velocidad: {speedup:.2f}x {'más rápido' if speedup > 1 else 'más lento'} (C++)")
    
    # Contains
    py_contains_per_item = py_results['contains_time'] / py_results['access_count']
    cpp_contains_per_item = cpp_results['contains_time'] / cpp_results['access_count']
    print(f"\n3. VERIFICACIÓN DE EXISTENCIA ({py_results['access_count']} verificaciones):")
    print(f"   Python Dict:    {py_results['contains_time']:.6f}s ({py_contains_per_item*1e6:.2f} μs/item)")
    print(f"   C++ SparseMap:  {cpp_results['contains_time']:.6f}s ({cpp_contains_per_item*1e6:.2f} μs/item)")
    if cpp_contains_per_item > 0:
        speedup = py_contains_per_item / cpp_contains_per_item
        print(f"   ⚡ Velocidad: {speedup:.2f}x {'más rápido' if speedup > 1 else 'más lento'} (C++)")
    
    # Keys
    print(f"\n4. ITERACIÓN DE CLAVES:")
    print(f"   Python Dict:    {py_results['keys_time']:.6f}s")
    print(f"   C++ SparseMap:  {cpp_results['keys_time']:.6f}s")
    if cpp_results['keys_time'] > 0:
        speedup = py_results['keys_time'] / cpp_results['keys_time']
        print(f"   ⚡ Velocidad: {speedup:.2f}x {'más rápido' if speedup > 1 else 'más lento'} (C++)")
    
    # Remove
    py_remove_per_item = py_results['remove_time'] / py_results['remove_count']
    cpp_remove_per_item = cpp_results['remove_time'] / cpp_results['remove_count']
    print(f"\n5. ELIMINACIÓN ({py_results['remove_count']} eliminaciones):")
    print(f"   Python Dict:    {py_results['remove_time']:.6f}s ({py_remove_per_item*1e6:.2f} μs/item)")
    print(f"   C++ SparseMap:  {cpp_results['remove_time']:.6f}s ({cpp_remove_per_item*1e6:.2f} μs/item)")
    if cpp_remove_per_item > 0:
        speedup = py_remove_per_item / cpp_remove_per_item
        print(f"   ⚡ Velocidad: {speedup:.2f}x {'más rápido' if speedup > 1 else 'más lento'} (C++)")
    
    print("\n" + "=" * 70)

def main():
    """Función principal"""
    print("=" * 70)
    print("🏎️  BENCHMARK: Almacenamiento de Tensores")
    print("   Python Dict vs C++ SparseMap (tensores nativos)")
    print("=" * 70)
    
    # Configuraciones de prueba
    test_configs = [
        (100, 4, "Pequeño (100 partículas)"),
        (1000, 4, "Mediano (1000 partículas)"),
        (10000, 4, "Grande (10000 partículas)"),
    ]
    
    all_results = []
    
    for num_particles, tensor_size, label in test_configs:
        print(f"\n{'=' * 70}")
        print(f"🔬 {label}")
        print(f"{'=' * 70}")
        
        # Warmup
        _ = benchmark_python_dict(10, tensor_size)
        if atheria_core.has_torch_support():
            _ = benchmark_cpp_sparsemap(10, tensor_size)
        
        # Ejecutar benchmarks
        py_results = benchmark_python_dict(num_particles, tensor_size)
        cpp_results = benchmark_cpp_sparsemap(num_particles, tensor_size)
        
        # Comparar
        print_comparison(py_results, cpp_results)
        
        all_results.append({
            'label': label,
            'num_particles': num_particles,
            'py': py_results,
            'cpp': cpp_results
        })
    
    # Benchmark de operaciones del motor
    print(f"\n{'=' * 70}")
    print(f"🔬 OPERACIONES DE MOTOR DISPERSO")
    print(f"{'=' * 70}")
    engine_results = benchmark_sparse_engine_operations(500, steps=5)
    print(f"\nMotor Python:  {engine_results['python_step_time']:.4f}s")
    print(f"Motor C++:     {engine_results['cpp_step_time']:.4f}s")
    if engine_results['cpp_step_time'] > 0:
        speedup = engine_results['python_step_time'] / engine_results['cpp_step_time']
        print(f"⚡ Velocidad: {speedup:.2f}x {'más rápido' if speedup > 1 else 'más lento'} (C++)")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    print("\n✅ Benchmarks completados")
    print(f"   Tests ejecutados: {len(test_configs)} configuraciones")
    print(f"   Tensores nativos C++: {'✅ Disponible' if atheria_core.has_torch_support() else '❌ No disponible'}")
    print("\n💡 Nota: Los resultados pueden variar según el hardware y carga del sistema.")
    print("=" * 70)

if __name__ == "__main__":
    main()

