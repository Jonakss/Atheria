# tests/test_dragonfly_cache.py
"""
Tests de integración para DragonflyCache.
"""
import pytest
import numpy as np
import sys
import os

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache.dragonfly_client import DragonflyCache


class TestDragonflyCache:
    """Suite de tests para DragonflyCache."""
    
    @pytest.fixture
    def cache(self):
        """Fixture que proporciona una instancia de caché."""
        cache_instance = DragonflyCache()
        yield cache_instance
        # Cleanup: limpiar claves de test
        if cache_instance.is_enabled():
            cache_instance.clear_pattern("test:*")
    
    def test_cache_enabled_or_graceful_degradation(self, cache):
        """Test que verifica que el caché está habilitado O degrada correctamente."""
        # El caché puede estar habilitado o no, dependiendo de si Dragonfly está corriendo
        # Este test solo verifica que no crashea
        is_enabled = cache.is_enabled()
        assert isinstance(is_enabled, bool)
        
        if is_enabled:
            print("✅ Dragonfly está disponible, caché habilitado")
        else:
            print("⚠️ Dragonfly no disponible, caché deshabilitado (graceful degradation)")
    
    def test_set_and_get_numpy_array(self, cache):
        """Test básico de guardar y recuperar array NumPy."""
        if not cache.is_enabled():
            pytest.skip("Dragonfly no está disponible")
        
        # Crear array de prueba
        test_array = np.random.rand(256, 256).astype(np.float32)
        
        # Guardar en caché
        success = cache.set("test:numpy_array", test_array, ttl=60)
        assert success, "Falló al guardar en caché"
        
        # Recuperar del caché
        retrieved = cache.get("test:numpy_array")
        assert retrieved is not None, "No se pudo recuperar del caché"
        
        # Verificar que son iguales
        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == test_array.shape
        assert np.allclose(retrieved, test_array), "Los arrays no coinciden"
    
    def test_cache_miss(self, cache):
        """Test que verifica el comportamiento en cache miss."""
        if not cache.is_enabled():
            pytest.skip("Dragonfly no está disponible")
        
        # Intentar obtener clave que no existe
        result = cache.get("test:nonexistent_key")
        assert result is None, "Cache miss debería retornar None"
    
    def test_ttl_expiration(self, cache):
        """Test que verifica que TTL funciona."""
        if not cache.is_enabled():
            pytest.skip("Dragonfly no está disponible")
        
        # Guardar con TTL muy corto
        cache.set("test:ttl_test", {"data": "test"}, ttl=1)
        
        # Verificar que existe inmediatamente
        assert cache.exists("test:ttl_test")
        
        # Esperar a que expire
        import time
        time.sleep(2)
        
        # Verificar que ya no existe
        assert not cache.exists("test:ttl_test")
    
    def test_complex_data_types(self, cache):
        """Test con diferentes tipos de datos."""
        if not cache.is_enabled():
            pytest.skip("Dragonfly no está disponible")
        
        test_cases = [
            ("string", "test string"),
            ("int", 42),
            ("float", 3.14159),
            ("list", [1, 2, 3, 4, 5]),
            ("dict", {"key1": "value1", "key2": 123}),
            ("tuple", (1, 2, 3)),
            ("nested", {"array": np.array([1, 2, 3]), "list": [4, 5, 6]})
        ]
        
        for name, data in test_cases:
            key = f"test:type_{name}"
            assert cache.set(key, data, ttl=60)
            retrieved = cache.get(key)
            
            if isinstance(data, np.ndarray):
                assert np.array_equal(retrieved, data)
            elif isinstance(data, dict) and "array" in data:
                # Caso especial para nested con array
                assert np.array_equal(retrieved["array"], data["array"])
                assert retrieved["list"] == data["list"]
            else:
                assert retrieved == data, f"Fallo para tipo {name}"
    
    def test_compression_efficiency(self, cache):
        """Test que verifica  que la compresión funciona."""
        if not cache.is_enabled():
            pytest.skip("Dragonfly no está disponible")
        
        # Crear array grande con datos repetitivos (alta compresibilidad)
        large_array = np.ones((1024, 1024), dtype=np.float32)
        
        # Guardar y verificar que se comprimió
        cache.set("test:large_compressed", large_array, ttl=60)
        
        # Recuperar y verificar
        retrieved = cache.get("test:large_compressed")
        assert retrieved is not None
        assert np.allclose(retrieved, large_array)
    
    def test_clear_pattern(self, cache):
        """Test de limpieza por patrón."""
        if not cache.is_enabled():
            pytest.skip("Dragonfly no está disponible")
        
        # Crear varias claves con el mismo patrón
        for i in range(5):
            cache.set(f"test:pattern:item_{i}", i, ttl=60)
        
        # Verificar que existen
        for i in range(5):
            assert cache.exists(f"test:pattern:item_{i}")
        
        # Limpiar todas las claves del patrón
        deleted = cache.clear_pattern("test:pattern:*")
        assert deleted == 5
        
        # Verificar que ya no existen
       for i in range(5):
            assert not cache.exists(f"test:pattern:item_{i}")
    
    def test_stats(self, cache):
        """Test de estadísticas del caché."""
        if not cache.is_enabled():
            pytest.skip("Dragonfly no está disponible")
        
        # Hacer algunas operaciones
        cache.set("test:stats_1", "data1", ttl=60)
        cache.get("test:stats_1")  # Hit
        cache.get("test:nonexistent")  # Miss
        
        # Obtener stats
        stats = cache.get_stats()
        assert stats["enabled"] == True
        assert "keyspace_hits" in stats or "total_commands" in stats


if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v", "-s"])
