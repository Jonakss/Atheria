#!/usr/bin/env python3
"""
Script de prueba avanzado para verificar el manejo de tensores PyTorch en C++.

Este script prueba que los tensores se pueden pasar desde Python a C++,
almacenarse correctamente, y recuperarse sin pérdida de datos.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import atheria_core

def test_torch_support():
    """Test 1: Verificar que el soporte de LibTorch está disponible"""
    print("=" * 60)
    print("Test 1: Verificación de Soporte LibTorch")
    print("=" * 60)
    
    has_support = atheria_core.has_torch_support()
    print(f"Soporte de LibTorch disponible: {has_support}")
    
    if not has_support:
        print("⚠️  LibTorch no está disponible. Los tests de tensores fallarán.")
        return False
    
    print("✅ Soporte de LibTorch disponible")
    print()
    return True

def test_tensor_storage():
    """Test 2: Almacenar y recuperar tensores"""
    print("=" * 60)
    print("Test 2: Almacenamiento y Recuperación de Tensores")
    print("=" * 60)
    
    if not atheria_core.has_torch_support():
        print("⚠️  Saltando test - LibTorch no disponible")
        return False
    
    smap = atheria_core.SparseMap()
    
    # Crear un tensor en Python
    tensor_orig = torch.ones(16, dtype=torch.float32)
    print(f"Tensor original: shape={tensor_orig.shape}, dtype={tensor_orig.dtype}")
    print(f"Valores: {tensor_orig[:5].tolist()}... (primeros 5)")
    
    # Crear coordenadas
    coord = atheria_core.Coord3D(0, 0, 0)
    
    # Insertar tensor
    smap.insert_tensor(coord, tensor_orig)
    print("✅ Tensor insertado")
    
    # Verificar que existe
    if smap.contains_coord(coord):
        print("✅ Coordenada existe en el mapa")
    else:
        print("❌ Coordenada no encontrada")
        return False
    
    # Recuperar tensor
    tensor_retrieved = smap.get_tensor(coord)
    
    if tensor_retrieved is None or tensor_retrieved.numel() == 0:
        print("❌ Tensor no recuperado o vacío")
        return False
    
    print(f"Tensor recuperado: shape={tensor_retrieved.shape}, dtype={tensor_retrieved.dtype}")
    print(f"Valores: {tensor_retrieved[:5].tolist()}... (primeros 5)")
    
    # Verificar que los datos son correctos
    if torch.allclose(tensor_orig, tensor_retrieved):
        print("✅ Datos verificados correctamente")
    else:
        print("❌ Los datos no coinciden")
        print(f"  Original: {tensor_orig[:5]}")
        print(f"  Recuperado: {tensor_retrieved[:5]}")
        return False
    
    print()
    return True

def test_multiple_tensors():
    """Test 3: Almacenar múltiples tensores"""
    print("=" * 60)
    print("Test 3: Múltiples Tensores")
    print("=" * 60)
    
    if not atheria_core.has_torch_support():
        print("⚠️  Saltando test - LibTorch no disponible")
        return False
    
    smap = atheria_core.SparseMap()
    
    # Crear varios tensores
    tensors = []
    coords = []
    
    for i in range(5):
        coord = atheria_core.Coord3D(i * 10, i * 20, i * 30)
        tensor = torch.randn(4, dtype=torch.float32)
        smap.insert_tensor(coord, tensor)
        tensors.append(tensor)
        coords.append(coord)
    
    print(f"✅ {len(tensors)} tensores insertados")
    
    # Verificar que todos existen
    for i, coord in enumerate(coords):
        if not smap.contains_coord(coord):
            print(f"❌ Coordenada {i} no encontrada")
            return False
    
    print("✅ Todas las coordenadas existen")
    
    # Recuperar y verificar todos
    for i, (coord, tensor_orig) in enumerate(zip(coords, tensors)):
        tensor_ret = smap.get_tensor(coord)
        if not torch.allclose(tensor_orig, tensor_ret):
            print(f"❌ Tensor {i} no coincide")
            return False
    
    print("✅ Todos los tensores se recuperaron correctamente")
    
    # Verificar coord_keys
    keys = smap.coord_keys()
    print(f"Coordenadas almacenadas: {len(keys)}")
    if len(keys) != len(coords):
        print(f"❌ Número de coordenadas incorrecto: esperado {len(coords)}, obtenido {len(keys)}")
        return False
    
    print("✅ Número de coordenadas correcto")
    print()
    return True

def test_tensor_operations():
    """Test 4: Operaciones con tensores"""
    print("=" * 60)
    print("Test 4: Operaciones con Tensores")
    print("=" * 60)
    
    if not atheria_core.has_torch_support():
        print("⚠️  Saltando test - LibTorch no disponible")
        return False
    
    smap = atheria_core.SparseMap()
    
    # Insertar tensor
    coord = atheria_core.Coord3D(100, 200, 300)
    tensor = torch.randn(8, dtype=torch.float32)
    smap.insert_tensor(coord, tensor)
    
    # Operación 1: Modificar tensor en Python y volver a insertar
    tensor_modified = tensor * 2.0
    smap.insert_tensor(coord, tensor_modified)
    
    tensor_ret = smap.get_tensor(coord)
    if not torch.allclose(tensor_modified, tensor_ret):
        print("❌ Tensor modificado no se guardó correctamente")
        return False
    
    print("✅ Modificación de tensor funciona")
    
    # Operación 2: Eliminar tensor
    smap.remove_coord(coord)
    if smap.contains_coord(coord):
        print("❌ Tensor no fue eliminado")
        return False
    
    print("✅ Eliminación de tensor funciona")
    
    # Operación 3: Limpiar todo
    smap.insert_tensor(coord, tensor)
    smap.clear()
    if smap.size() != 0:
        print("❌ clear() no funcionó correctamente")
        return False
    
    print("✅ clear() funciona correctamente")
    print()
    return True

def test_tensor_gradients():
    """Test 5: Manejo de gradientes (si aplica)"""
    print("=" * 60)
    print("Test 5: Manejo de Gradientes")
    print("=" * 60)
    
    if not atheria_core.has_torch_support():
        print("⚠️  Saltando test - LibTorch no disponible")
        return False
    
    smap = atheria_core.SparseMap()
    
    # Crear tensor con gradientes habilitados
    tensor = torch.ones(4, dtype=torch.float32, requires_grad=True)
    coord = atheria_core.Coord3D(0, 0, 0)
    
    # Insertar tensor
    smap.insert_tensor(coord, tensor)
    
    # Recuperar tensor
    tensor_ret = smap.get_tensor(coord)
    
    # Los tensores se copian, así que el requires_grad puede o no mantenerse
    # Depende de la implementación. Por ahora solo verificamos que se recupera.
    print(f"Tensor original requires_grad: {tensor.requires_grad}")
    print(f"Tensor recuperado requires_grad: {tensor_ret.requires_grad}")
    print(f"Tensor recuperado shape: {tensor_ret.shape}")
    
    if tensor_ret.numel() == tensor.numel():
        print("✅ Tensor con gradientes se maneja correctamente")
    else:
        print("❌ Tensor no se recuperó correctamente")
        return False
    
    print()
    return True

def test_complex_tensors():
    """Test 6: Tensores complejos (para estados cuánticos)"""
    print("=" * 60)
    print("Test 6: Tensores Complejos")
    print("=" * 60)
    
    if not atheria_core.has_torch_support():
        print("⚠️  Saltando test - LibTorch no disponible")
        return False
    
    smap = atheria_core.SparseMap()
    
    # Crear tensor complejo (como estados cuánticos en Atheria)
    real = torch.randn(4, dtype=torch.float32)
    imag = torch.randn(4, dtype=torch.float32)
    tensor_complex = torch.complex(real, imag)
    
    coord = atheria_core.Coord3D(50, 50, 50)
    
    print(f"Tensor complejo original: shape={tensor_complex.shape}, dtype={tensor_complex.dtype}")
    
    # Insertar tensor complejo
    smap.insert_tensor(coord, tensor_complex)
    
    # Recuperar
    tensor_ret = smap.get_tensor(coord)
    
    print(f"Tensor complejo recuperado: shape={tensor_ret.shape}, dtype={tensor_ret.dtype}")
    
    if torch.allclose(tensor_complex, tensor_ret):
        print("✅ Tensor complejo se almacena y recupera correctamente")
    else:
        print("❌ Tensor complejo no coincide")
        print(f"  Original real: {tensor_complex.real[:2]}")
        print(f"  Recuperado real: {tensor_ret.real[:2]}")
        return False
    
    print()
    return True

def main():
    """Función principal"""
    print("\n" + "=" * 60)
    print("🧪 Test Avanzado de Tensores PyTorch en C++")
    print("=" * 60)
    print()
    
    tests = [
        ("Verificación de Soporte LibTorch", test_torch_support),
        ("Almacenamiento y Recuperación", test_tensor_storage),
        ("Múltiples Tensores", test_multiple_tensors),
        ("Operaciones con Tensores", test_tensor_operations),
        ("Manejo de Gradientes", test_tensor_gradients),
        ("Tensores Complejos", test_complex_tensors),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Resumen
    print("=" * 60)
    print("Resumen de Tests")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Tests pasados: {passed}/{total}")
    
    if passed == total:
        print("✅ Todos los tests pasaron exitosamente!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) fallaron")
        return 1

if __name__ == "__main__":
    sys.exit(main())

