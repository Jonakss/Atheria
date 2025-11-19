#!/usr/bin/env python3
"""
Ejemplo de integración de atheria_core con el resto del proyecto Atheria.

Muestra cómo usar el núcleo C++ junto con los componentes Python existentes.
"""

import atheria_core
import torch
import numpy as np

def ejemplo_integracion_torch():
    """Ejemplo de integración con PyTorch"""
    print("=" * 60)
    print("Integración atheria_core con PyTorch")
    print("=" * 60)
    
    # Crear un tensor de PyTorch
    tensor = torch.randn(10, 10)
    
    # Usar SparseMap para almacenar índices activos (no-cero)
    sparse_indices = atheria_core.SparseMap()
    
    # Encontrar elementos no-cero y almacenarlos
    nonzero_count = 0
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            if abs(tensor[i, j].item()) > 0.5:
                key = i * tensor.shape[1] + j
                sparse_indices[key] = tensor[i, j].item()
                nonzero_count += 1
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Elementos no-cero encontrados: {nonzero_count}")
    print(f"Almacenados en SparseMap: {len(sparse_indices)}")
    
    # Recuperar valores y reconstruir tensor disperso
    recovered = torch.zeros_like(tensor)
    for key in sparse_indices.keys():
        i = key // tensor.shape[1]
        j = key % tensor.shape[1]
        recovered[i, j] = sparse_indices[key]
    
    print(f"Tensor reconstruido (shape): {recovered.shape}")
    print(f"Elementos no-cero reconstruidos: {(recovered != 0).sum().item()}")
    print()

def ejemplo_motor_fisica_disperso():
    """Ejemplo conceptual de motor de física disperso"""
    print("=" * 60)
    print("Concepto: Motor de Física Disperso")
    print("=" * 60)
    
    # Simular un estado cuántico disperso (como en Atheria)
    estado_cuantico = atheria_core.SparseMap()
    
    # Agregar amplitudes de probabilidad en posiciones específicas
    # (simulando un estado disperso en un espacio grande)
    posiciones_activas = [
        (100, 200, 0.5 + 0.3j),
        (150, 250, 0.3 + 0.4j),
        (200, 300, 0.2 + 0.1j),
    ]
    
    for x, y, amplitud in posiciones_activas:
        # Codificar posición y amplitud
        key = x << 16 | y
        # Para este ejemplo, almacenamos la magnitud
        estado_cuantico[key] = abs(amplitud)
    
    print(f"Estado cuántico disperso: {len(estado_cuantico)} posiciones activas")
    
    # Consultar una posición específica
    key = 100 << 16 | 200
    if key in estado_cuantico:
        print(f"Amplitud en (100, 200): {estado_cuantico[key]}")
    
    # Evolución temporal: actualizar amplitudes
    for key in estado_cuantico.keys():
        # Simular decaimiento
        estado_cuantico[key] *= 0.99
    
    print(f"Después de evolución: {len(estado_cuantico)} posiciones activas")
    print()

def ejemplo_benchmark_rendimiento():
    """Comparación de rendimiento con diccionarios Python"""
    print("=" * 60)
    print("Benchmark: SparseMap C++ vs dict Python")
    print("=" * 60)
    
    import time
    
    num_elementos = 100000
    
    # Test con SparseMap (C++)
    smap = atheria_core.SparseMap()
    start = time.time()
    for i in range(num_elementos):
        smap[i] = i * 1.5
    cpp_insert = time.time() - start
    
    start = time.time()
    for i in range(0, num_elementos, 100):
        _ = smap.get(i)
    cpp_access = time.time() - start
    
    # Test con dict Python
    py_dict = {}
    start = time.time()
    for i in range(num_elementos):
        py_dict[i] = i * 1.5
    py_insert = time.time() - start
    
    start = time.time()
    for i in range(0, num_elementos, 100):
        _ = py_dict.get(i)
    py_access = time.time() - start
    
    print(f"Insertar {num_elementos} elementos:")
    print(f"  SparseMap (C++): {cpp_insert:.4f}s")
    print(f"  dict (Python):   {py_insert:.4f}s")
    print(f"  Velocidad: {py_insert/cpp_insert:.2f}x más rápido" if cpp_insert > 0 else "  N/A")
    print()
    print(f"Acceso aleatorio (1000 accesos):")
    print(f"  SparseMap (C++): {cpp_access:.4f}s")
    print(f"  dict (Python):   {py_access:.4f}s")
    print(f"  Velocidad: {py_access/cpp_access:.2f}x más rápido" if cpp_access > 0 else "  N/A")
    print()

def main():
    """Función principal"""
    print("\n" + "=" * 60)
    print("🔗 Ejemplos de Integración atheria_core")
    print("=" * 60)
    print()
    
    try:
        ejemplo_integracion_torch()
        ejemplo_motor_fisica_disperso()
        ejemplo_benchmark_rendimiento()
        
        print("=" * 60)
        print("✅ Ejemplos de integración completados!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

