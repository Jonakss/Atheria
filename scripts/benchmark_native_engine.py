#!/usr/bin/env python3
"""
Benchmark comparativo: Motor Python vs Motor Nativo C++.

Este script compara el rendimiento entre:
1. Motor Python (sparse_engine.py)
2. Motor C++ con bindings básicos (sparse_engine_cpp.py)
3. Motor Nativo C++ con inferencia completa (Engine.step_native)
"""
import sys
from pathlib import Path
import time
import torch

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engines.sparse_engine import SparseQuantumEngine
from src.engines.sparse_engine_cpp import SparseQuantumEngineCpp
import atheria_core

def benchmark_engine_python(num_particles, steps, d_state=4):
    """Benchmark del motor Python puro"""
    print(f"\n📊 Motor Python (control)")
    
    engine = SparseQuantumEngine(None, d_state, 'cpu')
    
    # Agregar partículas
    start = time.time()
    for i in range(num_particles):
        state = torch.randn(d_state, dtype=torch.float32) * 0.1
        engine.add_particle((i*10, i*20, i*30), state)
    insert_time = time.time() - start
    
    # Ejecutar pasos
    start = time.time()
    for _ in range(steps):
        count = engine.step()
    step_time = time.time() - start
    
    print(f"   Inserción: {insert_time:.4f}s")
    print(f"   Pasos ({steps}): {step_time:.4f}s")
    print(f"   Partículas finales: {engine.get_matter_count()}")
    
    return {
        'name': 'Python',
        'insert_time': insert_time,
        'step_time': step_time,
        'particles': engine.get_matter_count()
    }

def benchmark_engine_cpp_bindings(num_particles, steps, d_state=4):
    """Benchmark del motor C++ con bindings básicos"""
    print(f"\n📊 Motor C++ (bindings básicos)")
    
    engine = SparseQuantumEngineCpp(None, d_state, 'cpu')
    
    # Agregar partículas
    start = time.time()
    for i in range(num_particles):
        state = torch.randn(d_state, dtype=torch.float32) * 0.1
        engine.add_particle((i*10, i*20, i*30), state)
    insert_time = time.time() - start
    
    # Ejecutar pasos
    start = time.time()
    for _ in range(steps):
        count = engine.step()
    step_time = time.time() - start
    
    print(f"   Inserción: {insert_time:.4f}s")
    print(f"   Pasos ({steps}): {step_time:.4f}s")
    print(f"   Partículas finales: {engine.get_matter_count()}")
    
    return {
        'name': 'C++ Bindings',
        'insert_time': insert_time,
        'step_time': step_time,
        'particles': engine.get_matter_count()
    }

def benchmark_engine_native(num_particles, steps, d_state=4):
    """Benchmark del motor nativo C++ con inferencia completa"""
    if not atheria_core.has_torch_support():
        print("\n⚠️  LibTorch no disponible, saltando motor nativo")
        return None
    
    print(f"\n📊 Motor Nativo C++ (inferencia completa)")
    
    engine = atheria_core.Engine(d_state, "cpu")
    
    # Agregar partículas
    start = time.time()
    for i in range(num_particles):
        coord = atheria_core.Coord3D(i*10, i*20, i*30)
        state = torch.randn(d_state, dtype=torch.complex64) * 0.1
        engine.add_particle(coord, state)
    insert_time = time.time() - start
    
    # Ejecutar pasos
    start = time.time()
    for _ in range(steps):
        count = engine.step_native()
    step_time = time.time() - start
    
    print(f"   Inserción: {insert_time:.4f}s")
    print(f"   Pasos ({steps}): {step_time:.4f}s")
    print(f"   Partículas finales: {engine.get_matter_count()}")
    
    return {
        'name': 'C++ Native',
        'insert_time': insert_time,
        'step_time': step_time,
        'particles': engine.get_matter_count()
    }

def main():
    """Función principal"""
    print("=" * 70)
    print("🏎️  BENCHMARK: Rendimiento de Motores Dispersos")
    print("=" * 70)
    
    # Configuraciones de prueba
    configs = [
        (100, 10, "Pequeño (100 partículas, 10 pasos)"),
        (500, 10, "Mediano (500 partículas, 10 pasos)"),
        (1000, 5, "Grande (1000 partículas, 5 pasos)"),
    ]
    
    all_results = []
    
    for num_particles, steps, label in configs:
        print(f"\n{'=' * 70}")
        print(f"🔬 {label}")
        print(f"{'=' * 70}")
        
        results = []
        
        # Test 1: Python
        try:
            result_py = benchmark_engine_python(num_particles, steps)
            results.append(result_py)
        except Exception as e:
            print(f"❌ Error en motor Python: {e}")
        
        # Test 2: C++ Bindings
        try:
            result_cpp = benchmark_engine_cpp_bindings(num_particles, steps)
            results.append(result_cpp)
        except Exception as e:
            print(f"❌ Error en motor C++ bindings: {e}")
        
        # Test 3: C++ Native
        try:
            result_native = benchmark_engine_native(num_particles, steps)
            if result_native:
                results.append(result_native)
        except Exception as e:
            print(f"❌ Error en motor C++ native: {e}")
        
        # Comparación
        if len(results) >= 2:
            print(f"\n📈 Comparación de Rendimiento:")
            base_time = results[0]['step_time']
            for result in results:
                speedup = base_time / result['step_time'] if result['step_time'] > 0 else 0
                status = "⚡" if speedup > 1 else "🐌"
                print(f"   {status} {result['name']:15} : {result['step_time']:.4f}s ({speedup:.2f}x vs Python)")
        
        all_results.append({
            'config': label,
            'results': results
        })
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    
    for config_data in all_results:
        print(f"\n{config_data['config']}:")
        for result in config_data['results']:
            print(f"   {result['name']:15} : {result['step_time']:.4f}s")
    
    print("\n💡 Observaciones:")
    print("   - Motor Nativo ejecuta TODO en C++ (sin marshaling)")
    print("   - Mejoras significativas esperadas con modelos TorchScript")
    print("   - Para uso completo, exporta un modelo con export_model_to_jit.py")
    print("=" * 70)

if __name__ == "__main__":
    main()

