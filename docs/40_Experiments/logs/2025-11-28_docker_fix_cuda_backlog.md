# 2025-11-28 - Fix: Docker Compose Compatibility & CUDA Optimization Backlog

**Contexto:** CUDA Out of Memory durante entrenamiento en GPU de 3.68 GiB y error de docker-compose al iniciar Dragonfly.

## Problema 1: CUDA Out of Memory

Durante entrenamiento con UNetUnitary (27.8M parámetros):
```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 3.68 GiB 
of which 11.75 MiB is free.
```

**Configuración actual:**
- Modelo: UNetUnitary
- Parámetros: 27.8M
- Grid: 64x64
- d_state: 14
- hidden_channels: 128
- qca_steps: 500

**Uso de memoria:** 3.56 GiB / 3.68 GiB (98%)

### Decisión: Backlog

Movido a backlog ya que requiere decisión sobre trade-offs:
1. **Mixed Precision (FP16)** - Reduce 50% memoria, posible pérdida de precisión
2. **Gradient Checkpointing** - Reduce 30-40% memoria, +30% tiempo de entrenamiento
3. **Reducir arquitectura** - hidden_channels 128→64, reduce 75% parámetros

**Documentación creada:**
- `docs/50_Backlog/CUDA_MEMORY_OPTIMIZATION.md` - Plan detallado
- `docs/10_core/ROADMAP_PHASE_5_BACKLOG.md` - Referencia en roadmap

---

## Problema 2: Docker Compose Incompatibilidad

**Error original:**
```
urllib3.exceptions.URLSchemeUnknown: Not supported URL scheme http+docker
```

**Causa raíz:**
- docker-compose v1.29.2 (antiguo) usa esquema `http+docker://`
- urllib3 v2.5.0 removió soporte para este esquema
- requests v2.32.5 depende de urllib3 v2.x

### Solución Aplicada

**1. Downgrade de librerías:**
```bash
pip install --user urllib3<2.0 requests<2.29
```

Resultado:
- urllib3: 2.5.0 → 1.26.20
- requests: 2.32.5 → 2.28.2

**Warnings generados:**
- `jupyterlab-server` y `jsonschema-path` requieren requests>=2.31
- Impacto mínimo: solo afecta notebooks externos (no crítico para entrenamiento)

---

## Problema 3: Dragonfly Memory Configuration

**Error inicial:**
```
E There are 16 threads, so 4.00GiB are required. Exiting...
```

**Causa:** Dragonfly con 16 threads (auto-detectados) necesitaba 4GB pero solo tenía 2GB configurados.

### Solución Aplicada

**Modificaciones en `docker-compose.yml`:**
```yaml
--maxmemory 4G          # Incrementado de 2G
--proactor_threads 4    # Limitado explícitamente
```

**Resultado:**
```
I Max memory limit is: 4.00GiB
I Running 4 io threads
```

**Estado final:** ✅ Dragonfly corriendo en puerto 6379

---

## Archivos Modificados

### Configuración
- `docker-compose.yml` - Aumentar memoria y limitar threads
- `requirements-docker-compat.txt` (nuevo) - Versiones compatibles

### Documentación
- `docs/50_Backlog/CUDA_MEMORY_OPTIMIZATION.md` (nuevo) - Plan de optimización de memoria
- `docs/10_core/ROADMAP_PHASE_5_BACKLOG.md` - Agregada sección de backlog

### Scripts
- `scripts/docker-compose-wrapper.sh` (nuevo) - Wrapper para DOCKER_HOST (no funcionó, alternativa)

---

## Lecciones Aprendidas

1. **Compatibilidad de versiones:** docker-compose v1.29.2 es obsoleto y vulnerable a cambios breaking en dependencias
2. **Dragonfly auto-detection:** Detecta todos los cores disponibles pero puede exceder memoria configurada
3. **GPU pequeñas:** Modelos grandes (>20M params) en GPUs <4GB requieren optimizaciones específicas

## Próximos Pasos (Backlog)

- [ ] Evaluar implementación de Mixed Precision (FP16) para entrenamiento
- [ ] Considerar upgrade a Docker Compose V2 (comando `docker compose` sin guión)
- [ ] Crear perfil "low_memory" para GPUs <4GB

---

**Referencias:**
- [[CUDA_MEMORY_OPTIMIZATION]] - Plan detallado de optimización
- [[UNET_UNITARY]] - Arquitectura del modelo
- [Docker Compose V1 Deprecation](https://docs.docker.com/compose/)

**Timestamp:** 2025-11-28 18:50 UTC-3  
**Duración:** ~40 minutos  
**Prioridad:** Alta (Docker), Media (CUDA optimization)
