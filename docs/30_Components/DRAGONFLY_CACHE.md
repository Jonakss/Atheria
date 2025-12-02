# Dragonfly Cache - Atheria

Sistema de caché distribuido de alta velocidad para estados cuánticos y checkpoints.

## ¿Qué es?

**Dragonfly** es un reemplazo de Redis 25x más rápido, compatible con la API de Redis. En Atheria, lo usamos para:

- ✅ Cachear estados cuánticos intermedios (cada 100 pasos)
- ✅ Recuperación instantánea de checkpoints (<50ms vs ~2s desde disco)
- ✅ Compartir estados entre procesos (entrenamiento + servidor)
- ✅ Time-travel sin recompute (saltar a cualquier paso cacheado)

## Inicio Rápido

### 1. Instalar Dependencias

```bash
# En el entorno virtual de Atheria
pip install redis>=5.0.0 zstandard>=0.21.0
```

### 2. Iniciar Dragonfly

```bash
# Desde la raíz del proyecto
docker-compose up -d dragonfly

# Verificar que está corriendo
docker ps | grep dragonfly
redis-cli ping  # Debe responder PONG
```

### 3. Usar en Código

El caché es **automático** y **transparente**:

```python
from src.cache import cache

# El caché se inicializa automáticamente (singleton)
# Si Dragonfly no está disponible, degrada gracefully

# Guardar estado
cache.set("state:exp_name:1000", state_array, ttl=7200)

# Recuperar estado
cached_state = cache.get("state:exp_name:1000")

# Verificar si existe
if cache.exists("state:exp_name:1000"):
    print("Estado en caché!")

# Limpiar estados antiguos
cache.clear_pattern("state:old_exp:*")
```

## Configuración

Variables de entorno (`.env` o export):

```bash
# Habilitar/deshabilitar caché
CACHE_ENABLED=true

# Host y puerto de Dragonfly
DRAGONFLY_HOST=localhost
DRAGONFLY_PORT=6379

# Intervalo de caché (cada N pasos)
CACHE_STATE_INTERVAL=100

# TTL por defecto (segundos)
CACHE_TTL=7200  # 2 horas
```

## Casos de Uso

### 1. Time-Travel en Inferencia

```python
# Saltar a paso 5000 sin recomputar
target_step = 5000
cached = cache.get(f"state:{exp_name}:{target_step}")

if cached is not None:
    motor.field = torch.from_numpy(cached).to(device)
    print(f"✅ Saltado a paso {target_step} desde caché!")
else:
    print(f"⚠️ Paso {target_step} no en caché, recomputando...")
    # ... recomputar
```

### 2. Compartir Estados entre Procesos

```python
# Proceso A (entrenamiento)
cache.set(f"latest_state:{exp_name}", current_state, ttl=3600)

# Proceso B (servidor de inferencia)
latest = cache.get(f"latest_state:{exp_name}")
if latest:
    print("✅ Estado más reciente recuperado del entrenamiento!")
```

### 3. Checkpoint Rápido

```python
# Guardar checkpoint en caché (además de disco)
checkpoint_data = {
    'model_state_dict': model.state_dict(),
    'episode': episode,
    'metrics': metrics
}

# Disco (permanente, lento)
torch.save(checkpoint_data, checkpoint_path)

# Caché (temporal, rápido)
cache.set(f"checkpoint:{exp_name}:latest", checkpoint_data, ttl=86400)

# Recuperar rápido
if cache.exists(f"checkpoint:{exp_name}:latest"):
    checkpoint = cache.get(f"checkpoint:{exp_name}:latest")  # <50ms!
```

### 4. Cache Buffering (Streaming)

El sistema de buffering desacopla la velocidad de simulación de la velocidad de visualización, permitiendo un flujo de datos constante al frontend.

**Flujo:**
1. **Productor (`DataProcessingService`)**: Empuja frames serializados a una lista en Dragonfly (`RPUSH`).
2. **Buffer (Dragonfly List)**: Almacena hasta `CACHE_STREAM_MAX_LEN` frames.
3. **Consumidor (`WebSocketService`)**: Consume frames (`LPOP`) a una tasa constante (`target_fps`) y los transmite.

**Configuración (`src/config.py`):**
```python
CACHE_BUFFERING_ENABLED = True
CACHE_STREAM_KEY = "simulation:stream"
CACHE_STREAM_MAX_LEN = 60
```

**Beneficios:**
- Evita saturación del frontend si la simulación es muy rápida.
- Suaviza la visualización (elimina "saltos").
- Maneja backpressure automáticamente (descarta frames antiguos si el buffer se llena).

## Monitoreo

### Ver Estadísticas

```python
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Total hits: {stats['keyspace_hits']}")
print(f"Total misses: {stats['keyspace_misses']}")
```

### Comandos Redis-CLI

```bash
# Conectar al caché
redis-cli

# Ver todas las claves
KEYS *

# Ver info
INFO stats

# Ver memoria usada
INFO memory

# Limpiar todo (¡CUIDADO!)
FLUSHALL
```

## Tests

```bash
# Ejecutar tests de integración
pytest tests/test_dragonfly_cache.py -v

# Con Dragonfly corriendo, todos deberían pasar ✅
# Sin Dragonfly, se saltarán (graceful degradation) ⏭️
```

## Performance

### Benchmarks

```python
import time
import numpy as np
from src.cache import cache

# Array 512x512
test_array = np.random.rand(512, 512).astype(np.float32)

# Test SET
start = time.time()
for i in range(1000):
    cache.set(f"bench:state_{i}", test_array, ttl=3600)
set_time = time.time() - start
print(f"SET: {set_time:.3f}s para 1000 estados ({1000/set_time:.0f} ops/s)")

# Test GET
start = time.time()
for i in range(1000):
    _ = cache.get(f"bench:state_{i}")
get_time = time.time() - start
print(f"GET: {get_time:.3f}s para 1000 estados ({1000/get_time:.0f} ops/s)")
```

**Resultados esperados** (hardware promedio):
- SET: ~2000 ops/s
- GET: ~5000 ops/s
- Latencia: <1ms por operación

## Troubleshooting

### Dragonfly no se conecta

```bash
# Verificar que está corriendo
docker ps | grep dragonfly

# Ver logs
docker logs atheria_dragonfly

# Reiniciar
docker-compose restart dragonfly
```

### Errores de memoria

```bash
# Ver uso de memoria
redis-cli INFO memory

# Limpiar caché antiguo
redis-cli FLUSHALL

# Ajustar maxmemory en docker-compose.yml
# --maxmemory 4G  (aumentar de 2G a 4G)
```

### Librerías faltantes

```python
# Si falta redis o zstandard, el caché se deshabilita automáticamente
# Ver warning en logs:
# "redis o zstandard no disponibles. Caché deshabilitado."

# Instalar:
pip install redis zstandard
```

## Próximos Pasos

- [ ] Integrar en motor QCA (`Aetheria_Motor`)
- [ ] Optimizar checkpoints en trainers
- [ ] Metrics dashboard (Grafana)
- [ ] Clustering para múltiples nodos

## Referencias

- [Dragonfly DB](https://www.dragonflydb.io/)
- [Redis Python Client](https://redis-py.readthedocs.io/)
- [Zstandard Compression](https://github.com/facebook/zstd)
