#!/usr/bin/env python3
"""
Benchmark comparativo de motores dispersos:
1. Python puro (sparse_engine.py)
2. C++ con diccionario auxiliar (sparse_engine_cpp.py) - V1
3. C++ con tensores nativos (sparse_engine_cpp_v2.py) - V2

Este script compara el rendimiento real de los motores en uso.
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
from src.engines.sparse_engine_cpp_v2 import SparseQuantumEngineCppV2

def benchmark_engine(engine_class, name, num_particles, steps, d_state=4):
    """Benchmark de un motor específico"""
    print(f"\n📊 {name}")
    
    # Crear motor
    engine = engine_class(None, d_state, 'cpu')
    
    # Información de almacenamiento
    if hasattr(engine, 'get_storage_info'):
        info = engine.get_storage_info()
        print(f"   Método: {info['method']}")
        print(f"   C++ nativo: {info.get('torch_support', False)}")
    
    # Agregar partículas
    start = time.time()
    for i in range(num_particles):
        state = torch.randn(d_state, dtype=torch.float32) * 0.1
        engine.add_particle((i*10, i*20, i*30), state)
    insert_time = time.time() - start
    
    print(f"   Inserción ({num_particles} partículas): {insert_time:.4f}s")
    
    # Ejecutar pasos
    start = time.time()
    for step in range(steps):
        count = engine.step()
    step_time = time.time() - start
    
    print(f"   Pasos ({steps} pasos): {step_time:.4f}s")
    print(f"   Partículas finales: {engine.get_matter_count()}")
    
    return {
        'name': name,
        'insert_time': insert_time,
        'step_time': step_time,
        'final_count': engine.get_matter_count()
    }

def main():
    """Función principal"""
    print("=" * 70)
    print("🏎️  BENCHMARK: Comparación de Motores Dispersos")
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
        
        # Test 1: Python puro
        try:
            result_py = benchmark_engine(
                SparseQuantumEngine,
                "Motor Python (control)",
                num_particles, steps
            )
            results.append(result_py)
        except Exception as e:
            print(f"❌ Error en motor Python: {e}")
        
        # Test 2: C++ V1 (con diccionario auxiliar)
        try:
            result_cpp_v1 = benchmark_engine(
                SparseQuantumEngineCpp,
                "Motor C++ V1 (auxiliary dict)",
                num_particles, steps
            )
            results.append(result_cpp_v1)
        except Exception as e:
            print(f"❌ Error en motor C++ V1: {e}")
        
        # Test 3: C++ V2 (tensores nativos)
        try:
            result_cpp_v2 = benchmark_engine(
                SparseQuantumEngineCppV2,
                "Motor C++ V2 (native tensors)",
                num_particles, steps
            )
            results.append(result_cpp_v2)
        except Exception as e:
            print(f"❌ Error en motor C++ V2: {e}")
        
        # Comparación
        if len(results) >= 2:
            print(f"\n📈 Comparación:")
            base_time = results[0]['step_time']
            for result in results:
                speedup = base_time / result['step_time'] if result['step_time'] > 0 else 0
                status = "⚡" if speedup > 1 else "🐌"
                print(f"   {status} {result['name']}: {result['step_time']:.4f}s ({speedup:.2f}x vs Python)")
        
        all_results.append({
            'config': label,
            'results': results
        })
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    print("\n💡 Observaciones:")
    print("   - C++ V2 usa almacenamiento nativo de tensores (sin dict auxiliar)")
    print("   - Overhead de bindings afecta operaciones pequeñas")
    print("   - Ventajas aparecerán en operaciones más complejas y vectorizadas")
    print("   - Arquitectura mejor preparada para optimizaciones futuras")
    print("=" * 70)

if __name__ == "__main__":
    main()

