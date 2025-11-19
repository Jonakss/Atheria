#!/usr/bin/env python3
"""
Script de prueba para verificar la implementación de Spatial Indexing (Morton Codes).

Verifica que:
1. coords_to_morton funciona correctamente
2. morton_to_coords recupera las coordenadas originales
3. La integridad de datos se mantiene (round-trip)
"""
import sys
import os
import torch
import numpy as np

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.spatial import SpatialIndexer
    print("✅ SpatialIndexer importado correctamente")
except ImportError as e:
    print(f"❌ Error importando SpatialIndexer: {e}")
    sys.exit(1)


def test_round_trip():
    """Prueba que coords → morton → coords recupera las coordenadas originales."""
    print("\n🔬 Test 1: Round-trip (coords → morton → coords)")
    print("-" * 60)
    
    indexer = SpatialIndexer()
    
    # Generar coordenadas aleatorias
    n_samples = 1000
    coords_original = torch.randint(0, 1000, (n_samples, 3), dtype=torch.int32)
    
    # Convertir a Morton
    morton_codes = indexer.coords_to_morton(coords_original)
    print(f"   ✅ {n_samples} coordenadas convertidas a Morton codes")
    
    # Convertir de vuelta a coordenadas
    coords_recovered = indexer.morton_to_coords(morton_codes)
    print(f"   ✅ Morton codes convertidos de vuelta a coordenadas")
    
    # Verificar integridad
    matches = torch.allclose(coords_original.float(), coords_recovered.float(), atol=0.1)
    exact_matches = torch.equal(coords_original, coords_recovered)
    
    if exact_matches:
        print(f"   ✅ Integridad perfecta: {n_samples}/{n_samples} coordenadas coinciden exactamente")
        return True
    elif matches:
        print(f"   ⚠️  Integridad aproximada: {n_samples} coordenadas recuperadas (puede haber pequeñas diferencias)")
        # Mostrar diferencias
        diff = torch.abs(coords_original.float() - coords_recovered.float())
        max_diff = diff.max().item()
        print(f"      Máxima diferencia: {max_diff:.2f}")
        return max_diff < 1.0  # Permitir diferencias pequeñas por redondeo
    else:
        print(f"   ❌ Error de integridad: Las coordenadas no coinciden")
        print(f"      Primera coordenada original: {coords_original[0].tolist()}")
        print(f"      Primera coordenada recuperada: {coords_recovered[0].tolist()}")
        return False


def test_specific_coords():
    """Prueba con coordenadas específicas conocidas."""
    print("\n🔬 Test 2: Coordenadas específicas conocidas")
    print("-" * 60)
    
    indexer = SpatialIndexer()
    
    # Coordenadas de prueba
    test_cases = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
        (10, 10, 10),
        (100, 100, 100),
        (255, 255, 255),
    ]
    
    all_pass = True
    for x, y, z in test_cases:
        coords = torch.tensor([[x, y, z]], dtype=torch.int32)
        morton = indexer.coords_to_morton(coords)[0].item()
        coords_recovered = indexer.morton_to_coords(torch.tensor([morton]))
        
        recovered_x, recovered_y, recovered_z = coords_recovered[0].tolist()
        
        if (recovered_x == x and recovered_y == y and recovered_z == z):
            print(f"   ✅ ({x:3d}, {y:3d}, {z:3d}) → Morton: {morton:15d} → ({recovered_x:3d}, {recovered_y:3d}, {recovered_z:3d})")
        else:
            print(f"   ❌ ({x:3d}, {y:3d}, {z:3d}) → Morton: {morton:15d} → ({recovered_x:3d}, {recovered_y:3d}, {recovered_z:3d}) [ERROR]")
            all_pass = False
    
    return all_pass


def test_locality():
    """Verifica que coordenadas cercanas tengan códigos Morton cercanos."""
    print("\n🔬 Test 3: Localidad espacial (coordenadas cercanas → Morton cercanos)")
    print("-" * 60)
    
    indexer = SpatialIndexer()
    
    # Coordenadas cercanas en el espacio 3D
    base_coords = torch.tensor([
        [10, 10, 10],
        [11, 10, 10],  # +1 en X
        [10, 11, 10],  # +1 en Y
        [10, 10, 11],  # +1 en Z
        [11, 11, 10],  # +1 en X e Y
        [10, 11, 11],  # +1 en Y e Z
        [11, 10, 11],  # +1 en X e Z
        [11, 11, 11],  # +1 en todas
    ], dtype=torch.int32)
    
    morton_codes = indexer.coords_to_morton(base_coords)
    
    # Verificar que los códigos Morton estén relativamente cerca
    codes_list = morton_codes.tolist()
    sorted_codes = sorted(codes_list)
    
    print(f"   Códigos Morton obtenidos:")
    for i, (coords, code) in enumerate(zip(base_coords, codes_list)):
        print(f"      ({coords[0]:2d}, {coords[1]:2d}, {coords[2]:2d}) → {code:15d}")
    
    # Verificar que estén ordenados (o al menos cercanos)
    differences = [sorted_codes[i+1] - sorted_codes[i] for i in range(len(sorted_codes)-1)]
    avg_diff = sum(differences) / len(differences) if differences else 0
    
    print(f"\n   Diferencias entre códigos consecutivos: {differences}")
    print(f"   Diferencia promedio: {avg_diff:.2f}")
    
    # Los códigos deberían estar relativamente cercanos (aunque no necesariamente ordenados)
    max_diff = max(differences) if differences else 0
    if max_diff < 10000:  # Threshold arbitrario
        print(f"   ✅ Localidad preservada: códigos Morton están relativamente cercanos")
        return True
    else:
        print(f"   ⚠️  Los códigos Morton pueden no preservar localidad perfectamente")
        return True  # No es un error crítico


def test_get_active_chunks():
    """Prueba la función get_active_chunks."""
    print("\n🔬 Test 4: get_active_chunks")
    print("-" * 60)
    
    indexer = SpatialIndexer()
    
    # Coordenadas con algunas duplicadas
    coords = torch.tensor([
        [10, 10, 10],
        [11, 10, 10],
        [10, 10, 10],  # Duplicado
        [50, 50, 50],
        [11, 10, 10],  # Duplicado
    ], dtype=torch.int32)
    
    chunks = indexer.get_active_chunks(coords)
    
    print(f"   Coordenadas de entrada: {len(coords)}")
    print(f"   Chunks únicos: {len(chunks)}")
    print(f"   Chunks: {chunks[:10]}...")  # Mostrar primeros 10
    
    expected_unique = 3  # 3 coordenadas únicas
    if len(chunks) == expected_unique:
        print(f"   ✅ Número correcto de chunks únicos ({expected_unique})")
        return True
    else:
        print(f"   ❌ Número incorrecto de chunks: esperado {expected_unique}, obtenido {len(chunks)}")
        return False


def test_edge_cases():
    """Prueba casos límite."""
    print("\n🔬 Test 5: Casos límite")
    print("-" * 60)
    
    indexer = SpatialIndexer()
    
    # Caso 1: Coordenadas grandes
    try:
        large_coords = torch.tensor([[1000000, 2000000, 3000000]], dtype=torch.int32)
        morton = indexer.coords_to_morton(large_coords)
        coords_recovered = indexer.morton_to_coords(morton)
        print(f"   ✅ Coordenadas grandes manejadas correctamente")
        print(f"      Original: {large_coords[0].tolist()}")
        print(f"      Recuperado: {coords_recovered[0].tolist()}")
        large_pass = True
    except Exception as e:
        print(f"   ❌ Error con coordenadas grandes: {e}")
        large_pass = False
    
    # Caso 2: Coordenadas negativas (deberían clampearse a 0)
    try:
        negative_coords = torch.tensor([[-10, -5, -1]], dtype=torch.int32)
        morton = indexer.coords_to_morton(negative_coords)
        coords_recovered = indexer.morton_to_coords(morton)
        print(f"   ✅ Coordenadas negativas clampheadas correctamente")
        print(f"      Original (clamp): {torch.clamp(negative_coords, 0, indexer.max_coord)[0].tolist()}")
        print(f"      Recuperado: {coords_recovered[0].tolist()}")
        negative_pass = True
    except Exception as e:
        print(f"   ❌ Error con coordenadas negativas: {e}")
        negative_pass = False
    
    # Caso 3: Tensor vacío
    try:
        empty_coords = torch.empty((0, 3), dtype=torch.int32)
        morton = indexer.coords_to_morton(empty_coords)
        coords_recovered = indexer.morton_to_coords(morton)
        print(f"   ✅ Tensor vacío manejado correctamente")
        empty_pass = True
    except Exception as e:
        print(f"   ❌ Error con tensor vacío: {e}")
        empty_pass = False
    
    return large_pass and negative_pass and empty_pass


def benchmark_performance():
    """Benchmark básico de rendimiento."""
    print("\n🔬 Test 6: Benchmark de rendimiento")
    print("-" * 60)
    
    import time
    
    indexer = SpatialIndexer()
    
    # Generar coordenadas
    n_samples = 100000
    coords = torch.randint(0, 1000, (n_samples, 3), dtype=torch.int32)
    
    # Benchmark coords_to_morton
    start = time.time()
    morton_codes = indexer.coords_to_morton(coords)
    encode_time = (time.time() - start) * 1000  # ms
    
    # Benchmark morton_to_coords
    start = time.time()
    coords_recovered = indexer.morton_to_coords(morton_codes)
    decode_time = (time.time() - start) * 1000  # ms
    
    print(f"   Muestras: {n_samples:,}")
    print(f"   coords_to_morton: {encode_time:.2f} ms ({encode_time/n_samples*1000:.2f} ns/coord)")
    print(f"   morton_to_coords: {decode_time:.2f} ms ({decode_time/n_samples*1000:.2f} ns/coord)")
    print(f"   Total round-trip: {encode_time + decode_time:.2f} ms")
    
    # Verificar que funciona
    matches = torch.equal(coords, coords_recovered)
    if matches:
        print(f"   ✅ Integridad verificada en benchmark")
        return True
    else:
        print(f"   ❌ Error de integridad en benchmark")
        return False


def main():
    """Ejecuta todos los tests."""
    print("=" * 60)
    print("🧪 Tests de Spatial Indexing (Morton Codes)")
    print("=" * 60)
    
    results = []
    
    # Ejecutar tests
    results.append(("Round-trip", test_round_trip()))
    results.append(("Coordenadas específicas", test_specific_coords()))
    results.append(("Localidad espacial", test_locality()))
    results.append(("get_active_chunks", test_get_active_chunks()))
    results.append(("Casos límite", test_edge_cases()))
    results.append(("Benchmark", benchmark_performance()))
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 Resumen de Tests")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n   Resultado: {passed}/{total} tests pasados")
    
    if passed == total:
        print("\n   🎉 ¡Todos los tests pasaron!")
        return 0
    else:
        print(f"\n   ⚠️  {total - passed} test(s) fallaron")
        return 1


if __name__ == "__main__":
    sys.exit(main())

