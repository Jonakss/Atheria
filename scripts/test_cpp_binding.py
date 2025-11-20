#!/usr/bin/env python3
"""
Script de prueba para verificar que los bindings C++ funcionan correctamente.
Este es el "Hello World" de la Fase 2.
"""
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_add_function():
    """Prueba la función add() que es el Hello World."""
    print("=" * 60)
    print("🧪 TEST: Función add() - Hello World de Fase 2")
    print("=" * 60)
    
    try:
        import atheria_core
        print("✅ Módulo atheria_core importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando atheria_core: {e}")
        print("   El módulo no está compilado. Ejecuta:")
        print("   python setup.py build_ext --inplace")
        return False
    
    # Probar función add
    try:
        result = atheria_core.add(5, 3)
        expected = 8
        if result == expected:
            print(f"✅ add(5, 3) = {result} (esperado: {expected})")
        else:
            print(f"❌ add(5, 3) = {result} (esperado: {expected})")
            return False
    except Exception as e:
        print(f"❌ Error ejecutando add(): {e}")
        return False
    
    # Probar verificación de LibTorch
    try:
        has_torch = atheria_core.has_torch_support()
        print(f"✅ LibTorch disponible: {has_torch}")
    except Exception as e:
        print(f"⚠️  Error verificando LibTorch: {e}")
    
    return True

def test_coord3d():
    """Prueba la estructura Coord3D."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Estructura Coord3D")
    print("=" * 60)
    
    try:
        import atheria_core
        
        # Crear coordenada
        coord = atheria_core.Coord3D(10, 20, 30)
        print(f"✅ Coord3D creado: {coord}")
        
        # Verificar valores
        if coord.x == 10 and coord.y == 20 and coord.z == 30:
            print(f"   Valores correctos: x={coord.x}, y={coord.y}, z={coord.z}")
        else:
            print(f"❌ Valores incorrectos: x={coord.x}, y={coord.y}, z={coord.z}")
            return False
        
        # Modificar valores
        coord.x = 100
        coord.y = 200
        coord.z = 300
        if coord.x == 100 and coord.y == 200 and coord.z == 300:
            print(f"✅ Modificación exitosa: {coord}")
        else:
            print(f"❌ Error modificando coordenadas")
            return False
            
    except Exception as e:
        print(f"❌ Error probando Coord3D: {e}")
        return False
    
    return True

def test_sparse_map():
    """Prueba la clase SparseMap básica."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Clase SparseMap (operaciones básicas)")
    print("=" * 60)
    
    try:
        import atheria_core
        
        # Crear mapa
        sparse_map = atheria_core.SparseMap()
        print("✅ SparseMap creado")
        
        # Verificar que está vacío
        if sparse_map.empty():
            print(f"✅ Mapa vacío inicialmente: size={sparse_map.size()}")
        else:
            print(f"❌ Mapa debería estar vacío: size={sparse_map.size()}")
            return False
        
        # Insertar valores
        sparse_map.insert(1, 10.5)
        sparse_map.insert(2, 20.7)
        sparse_map.insert(3, 30.9)
        print(f"✅ Insertados 3 elementos: size={sparse_map.size()}")
        
        if sparse_map.size() != 3:
            print(f"❌ Tamaño incorrecto: {sparse_map.size()} (esperado: 3)")
            return False
        
        # Verificar contains
        if sparse_map.contains(1) and sparse_map.contains(2) and sparse_map.contains(3):
            print("✅ contains() funciona correctamente")
        else:
            print("❌ contains() falló")
            return False
        
        # Obtener valores
        val1 = sparse_map.get(1)
        val2 = sparse_map.get(2)
        val3 = sparse_map.get(3)
        
        if abs(val1 - 10.5) < 1e-9 and abs(val2 - 20.7) < 1e-9 and abs(val3 - 30.9) < 1e-9:
            print(f"✅ get() funciona: {val1}, {val2}, {val3}")
        else:
            print(f"❌ get() falló: {val1}, {val2}, {val3}")
            return False
        
        # Probar operadores Python
        if 1 in sparse_map:
            print("✅ Operador 'in' funciona")
        else:
            print("❌ Operador 'in' falló")
            return False
        
        # Probar acceso con []
        if abs(sparse_map[1] - 10.5) < 1e-9:
            print("✅ Operador [] funciona")
        else:
            print("❌ Operador [] falló")
            return False
        
        # Probar len()
        if len(sparse_map) == 3:
            print(f"✅ len() funciona: {len(sparse_map)}")
        else:
            print(f"❌ len() falló: {len(sparse_map)}")
            return False
        
        # Eliminar elemento
        sparse_map.remove(2)
        if sparse_map.size() == 2 and not sparse_map.contains(2):
            print("✅ remove() funciona correctamente")
        else:
            print(f"❌ remove() falló: size={sparse_map.size()}")
            return False
        
        # Limpiar
        sparse_map.clear()
        if sparse_map.empty():
            print("✅ clear() funciona correctamente")
        else:
            print("❌ clear() falló")
            return False
        
    except Exception as e:
        print(f"❌ Error probando SparseMap: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Ejecuta todos los tests."""
    print("\n" + "🚀 " + "=" * 58)
    print("   INICIANDO TESTS DE FASE 2: Motor Nativo C++")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Función add (Hello World)
    results.append(("add() - Hello World", test_add_function()))
    
    # Test 2: Coord3D
    results.append(("Coord3D", test_coord3d()))
    
    # Test 3: SparseMap básico
    results.append(("SparseMap básico", test_sparse_map()))
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE TESTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"✅ TODOS LOS TESTS PASARON ({passed}/{total})")
        print("🎉 Fase 2 - Hello World COMPLETADO!")
        return 0
    else:
        print(f"❌ ALGUNOS TESTS FALLARON ({passed}/{total})")
        return 1

if __name__ == "__main__":
    sys.exit(main())
