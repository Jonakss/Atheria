## 2024-12-20 - Problemas Críticos Motor Nativo Identificados

### Contexto
Se identificaron **dos problemas críticos** con el motor nativo C++:
1. **Cuelgue/Bloqueo**: El motor nativo se queda bloqueado durante la simulación
2. **Lentitud Extrema**: El motor nativo se pone muy lento en tiempo real

### Problemas Identificados

#### 1. Motor Nativo se Cuelga/Bloquea

**Síntoma:**
- El motor nativo se queda bloqueado durante la simulación
- No responde a comandos de pausa inmediatamente
- Requiere matar el proceso para detener

**Causa Raíz:**
- `step_native()` en C++ es bloqueante y no verifica pausa
- `_update_dense_state_from_sparse()` se ejecuta en cada paso y puede tomar mucho tiempo (65,536 coordenadas)
- No hay verificación de pausa durante la ejecución

**Ubicación:**
- `src/cpp_core/src/sparse_engine.cpp:71` - `step_native()` es bloqueante
- `src/engines/native_engine_wrapper.py:283` - `_update_dense_state_from_sparse()` se ejecuta en cada paso
- `src/pipelines/pipeline_server.py:257` - No hay verificación de pausa durante `evolve_internal_state()`

#### 2. Lentitud Extrema en Tiempo Real

**Síntoma:**
- El motor nativo se pone muy lento en tiempo real
- FPS cae dramáticamente
- UI se congela

**Causa Raíz:**
- Conversión completa en cada paso: itera sobre **todo el grid** (256x256 = **65,536 coordenadas**)
- 65,536 llamadas a `get_state_at()` en cada paso
- Overhead Python↔C++ × 65,536 = **MUY COSTOSO**

**Análisis:**
- Grid 256x256 = 65,536 coordenadas
- En cada paso: 65,536 llamadas a `get_state_at()`
- Cada llamada: overhead Python↔C++ (aproximadamente 10-50μs)
- **Total:** ~650ms - 3.2 segundos POR PASO solo en conversión

### Soluciones Propuestas

#### Solución 1: Lazy Conversion (Prioridad Alta)
- Solo convertir cuando se necesita visualizar
- Marcar estado como "stale" después de `evolve_internal_state()`
- Convertir solo cuando se llama `get_dense_state()`

#### Solución 2: ROI para Conversión (Prioridad Alta)
- Solo convertir región visible
- Reducir de 65,536 a ~10,000-20,000 coordenadas (si ROI es pequeño)
- 3-5x más rápido dependiendo del tamaño de ROI

#### Solución 3: Verificación de Pausa Durante Conversión (Prioridad Alta)
- Permitir pausa inmediata durante conversión
- Verificar pausa cada batch (1000 coordenadas)

#### Solución 4: Batch Conversion en C++ (Prioridad Media)
- Reducir overhead Python↔C++
- Agregar método `get_state_batch()` que obtiene múltiples coordenadas en una llamada
- 10-50x más rápido que llamadas individuales

### Archivos Afectados

1. **`src/engines/native_engine_wrapper.py`**
   - `evolve_internal_state()` - Ejecuta conversión en cada paso
   - `_update_dense_state_from_sparse()` - Conversión completa sobre todo el grid

2. **`src/pipelines/pipeline_server.py`**
   - `simulation_loop()` - No verifica pausa durante `evolve_internal_state()`

3. **`src/cpp_core/src/sparse_engine.cpp`**
   - `step_native()` - Es bloqueante y no verifica pausa

### Estado
🔴 **CRÍTICO - Pendiente de Implementación**

**Referencias:**
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES]] - Documentación detallada de problemas
- [[PENDING_TASKS]] - Lista completa de tareas pendientes
- `src/engines/native_engine_wrapper.py:271-372` - Código problemático

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
