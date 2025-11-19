#!/usr/bin/env python3
"""
Script de verificación para el binding C++ de Atheria Core.

Este script verifica que:
1. El módulo atheria_core puede ser importado
2. La función add() funciona correctamente
3. La clase SparseMap puede ser instanciada y usada
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path para imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_import():
    """Test 1: Importar el módulo"""
    print("Test 1: Importando módulo atheria_core...")
    try:
        import atheria_core
        print(f"  ✅ Módulo importado exitosamente")
        print(f"  📦 Ubicación: {atheria_core.__file__}")
        return atheria_core
    except ImportError as e:
        print(f"  ❌ Error al importar: {e}")
        print("\n💡 Sugerencia: Asegúrate de haber compilado el módulo:")
        print("   pip install -e .")
        sys.exit(1)

def test_add_function(atheria_core):
    """Test 2: Función add()"""
    print("\nTest 2: Probando función add()...")
    try:
        result = atheria_core.add(5, 3)
        expected = 8
        assert result == expected, f"Esperado {expected}, obtuvo {result}"
        print(f"  ✅ add(5, 3) = {result}")
        
        # Test con números negativos
        result2 = atheria_core.add(-10, 20)
        assert result2 == 10, f"Esperado 10, obtuvo {result2}"
        print(f"  ✅ add(-10, 20) = {result2}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_sparse_map(atheria_core):
    """Test 3: Clase SparseMap"""
    print("\nTest 3: Probando clase SparseMap...")
    try:
        # Crear instancia
        smap = atheria_core.SparseMap()
        print("  ✅ SparseMap instanciado")
        
        # Test inicial
        assert smap.empty(), "Mapa debería estar vacío inicialmente"
        assert smap.size() == 0, "Tamaño debería ser 0"
        print("  ✅ Estado inicial correcto (vacío)")
        
        # Insertar valores
        smap.insert(1, 10.5)
        smap.insert(2, 20.3)
        smap.insert(3, 30.7)
        print("  ✅ Valores insertados")
        
        # Verificar tamaño
        assert smap.size() == 3, f"Tamaño debería ser 3, obtuvo {smap.size()}"
        print(f"  ✅ Tamaño correcto: {smap.size()}")
        
        # Verificar contains
        assert smap.contains(1), "Debería contener clave 1"
        assert not smap.contains(999), "No debería contener clave 999"
        print("  ✅ contains() funciona correctamente")
        
        # Verificar get
        assert abs(smap.get(1) - 10.5) < 1e-9, "Valor incorrecto para clave 1"
        assert abs(smap.get(999, 0.0) - 0.0) < 1e-9, "Valor por defecto incorrecto"
        print("  ✅ get() funciona correctamente")
        
        # Test de acceso con []
        assert 1 in smap, "Clave 1 debería estar usando __contains__"
        value = smap[1]
        assert abs(value - 10.5) < 1e-9, "Acceso con [] falló"
        print("  ✅ Acceso con [] funciona")
        
        # Test de asignación con []
        smap[4] = 40.9
        assert abs(smap.get(4) - 40.9) < 1e-9, "Asignación con [] falló"
        print("  ✅ Asignación con [] funciona")
        
        # Test de eliminación
        smap.remove(2)
        assert not smap.contains(2), "Clave 2 debería haber sido eliminada"
        assert smap.size() == 3, "Tamaño debería ser 3 (1, 3, 4)"
        print("  ✅ remove() funciona correctamente")
        
        # Test de keys y values
        keys = smap.keys()
        values = smap.values()
        assert len(keys) == 3, f"Debería haber 3 claves, obtuvo {len(keys)}"
        assert len(values) == 3, f"Debería haber 3 valores, obtuvo {len(values)}"
        print(f"  ✅ keys() y values() funcionan: {keys}, {values}")
        
        # Test de clear
        smap.clear()
        assert smap.empty(), "Mapa debería estar vacío después de clear()"
        assert smap.size() == 0, "Tamaño debería ser 0"
        print("  ✅ clear() funciona correctamente")
        
        # Test de __repr__
        repr_str = repr(smap)
        assert "SparseMap" in repr_str, "repr() debería contener 'SparseMap'"
        print(f"  ✅ __repr__ funciona: {repr_str}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("=" * 60)
    print("🧪 Verificación del Binding C++ de Atheria Core")
    print("=" * 60)
    
    # Test 1: Import
    atheria_core = test_import()
    
    # Test 2: Función add
    if not test_add_function(atheria_core):
        print("\n❌ Tests fallaron en función add()")
        sys.exit(1)
    
    # Test 3: Clase SparseMap
    if not test_sparse_map(atheria_core):
        print("\n❌ Tests fallaron en clase SparseMap")
        sys.exit(1)
    
    # Éxito
    print("\n" + "=" * 60)
    print("✅ Todos los tests pasaron exitosamente!")
    print("✅ C++ Binding Exitoso")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

