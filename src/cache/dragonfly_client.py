# src/cache/dragonfly_client.py
"""
Cliente Dragonfly para caché de estados cuánticos y checkpoints.
Compatible con Redis, usa compresión zstd para eficiencia.
"""
import logging
import pickle
import os
from typing import Any, Optional

try:
    import redis
    import zstandard as zstd
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logging.warning("redis o zstandard no disponibles. Caché deshabilitado.")


class DragonflyCache:
    """
    Cliente singleton para Dragonfly (Redis-compatible).
    Proporciona caché de alta velocidad para estados cuánticos y checkpoints.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        host: str = None,
        port: int = None,
        enabled: bool = None
    ):
        # Evitar re-inicialización
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # Configuración desde env o defaults
        self.host = host or os.getenv('DRAGONFLY_HOST', 'localhost')
        self.port = port or int(os.getenv('DRAGONFLY_PORT', '6379'))
        self.enabled = enabled if enabled is not None else \
                      os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        
        self.client = None
        self.compressor = None
        
        if not CACHE_AVAILABLE:
            self.enabled = False
            logging.info("🔴 Dragonfly: Librerías no disponibles. Caché deshabilitado.")
            return
        
        if not self.enabled:
            logging.info("🔴 Dragonfly: Caché deshabilitado por configuración.")
            return
        
        # Intentar conectar
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                decode_responses=False,  # Trabajar con bytes
                socket_timeout=2,
                socket_connect_timeout=2
            )
            # Verificar conexión
            self.client.ping()
            
            # Inicializar compresor zstd
            self.compressor = zstd.ZstdCompressor(level=3)  # Nivel 3 = balance velocidad/compresión
            self.decompressor = zstd.ZstdDecompressor()
            
            self.enabled = True
            logging.info(f"✅ Dragonfly: Conectado a {self.host}:{self.port}")
            
        except Exception as e:
            self.enabled = False
            self.client = None
            logging.warning(f"⚠️ Dragonfly: No se pudo conectar ({e}). Caché deshabilitado.")
    
    def is_enabled(self) -> bool:
        """Verifica si el caché está habilitado y funcional."""
        return self.enabled
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Guarda un valor en caché con TTL (Time To Live).
        
        Args:
            key: Clave única
            value: Valor a guardar (será serializado con pickle + zstd)
            ttl: Tiempo de vida en segundos (default: 1 hora)
        
        Returns:
            True si se guardó exitosamente, False si no
        """
        if not self.enabled or self.client is None:
            return False
        
        try:
            # Serializar con pickle
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Comprimir con zstd
            compressed = self.compressor.compress(serialized)
            
            # Guardar en Dragonfly con TTL
            self.client.setex(key, ttl, compressed)
            
            logging.debug(f"📦 Cache SET: {key} ({len(serialized)} → {len(compressed)} bytes, TTL={ttl}s)")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error guardando en caché '{key}': {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Recupera un valor del caché.
        
        Args:
            key: Clave única
        
        Returns:
            Valor deserializado o None si no existe/falló
        """
        if not self.enabled or self.client is None:
            return None
        
        try:
            # Obtener datos comprimidos
            compressed = self.client.get(key)
            
            if compressed is None:
                logging.debug(f"🔍 Cache MISS: {key}")
                return None
            
            # Descomprimir
            serialized = self.decompressor.decompress(compressed)
            
            # Deserializar
            value = pickle.loads(serialized)
            
            logging.debug(f"✅ Cache HIT: {key} ({len(compressed)} bytes)")
            return value
            
        except Exception as e:
            logging.error(f"❌ Error recuperando de caché '{key}': {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Elimina una clave del caché."""
        if not self.enabled or self.client is None:
            return False
        
        try:
            self.client.delete(key)
            logging.debug(f"🗑️ Cache DELETE: {key}")
            return True
        except Exception as e:
            logging.error(f"❌ Error eliminando de caché '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Verifica si una clave existe en el caché."""
        if not self.enabled or self.client is None:
            return False
        
        try:
            return bool(self.client.exists(key))
        except:
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Elimina todas las claves que coincidan con un patrón.
        
        Args:
            pattern: Patrón de Redis (ej: "state:exp123:*")
        
        Returns:
            Número de claves eliminadas
        """
        if not self.enabled or self.client is None:
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logging.info(f"🗑️ Cache: Eliminadas {deleted} claves con patrón '{pattern}'")
                return deleted
            return 0
        except Exception as e:
            logging.error(f"❌ Error limpiando patrón '{pattern}': {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas del caché."""
        if not self.enabled or self.client is None:
            return {"enabled": False}
        
        try:
            info = self.client.info('stats')
            return {
                "enabled": True,
                "total_commands": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "hit_rate": info.get('keyspace_hits', 0) / 
                           max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
            }
        except Exception as e:
            logging.error(f"❌ Error obteniendo stats: {e}")
            return {"enabled": True, "error": str(e)}


# Singleton global
cache = DragonflyCache()
