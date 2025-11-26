## 2025-01-21 - Corrección Fundamental: Generación de Estado Inicial según Ley M

### Contexto
El usuario reportó que el motor nativo cargaba correctamente pero los comandos (ejecutar, cargar otro modelo, descargar) no funcionaban. Al investigar, se descubrió un problema más fundamental: **las partículas se estaban agregando manualmente como un hack, en lugar de emerger del modelo cuántico (ley M)**.

### Problema Identificado
1. **Hack de inicialización**: El motor nativo usaba `add_initial_particles()` para agregar partículas aleatorias manualmente, en lugar de generar el estado inicial según `INITIAL_STATE_MODE_INFERENCE` (como lo hace el motor Python).
2. **Inconsistencia con ley M**: Las partículas deberían emerger del estado cuántico generado por el modelo, no agregarse manualmente.
3. **Logging insuficiente**: Los comandos WebSocket no tenían logging suficiente para diagnosticar problemas de comunicación.

### Solución Implementada

#### 1. Generación Correcta de Estado Inicial ✅

**Archivo:** `src/engines/native_engine_wrapper.py`

**Cambios:**
- `__init__()` ahora genera `QuantumState` con `initial_mode` desde `cfg.INITIAL_STATE_MODE_INFERENCE` (igual que el motor Python).
- Soporta grid scaling: si `training_grid_size < inference_grid_size`, replica el estado base.
- Llama automáticamente a `_initialize_native_state_from_dense()` después de generar el estado denso.

**Nuevo método: `_initialize_native_state_from_dense()`**
- Convierte estado denso inicial → formato disperso del motor nativo.
- Respeta `INITIAL_STATE_MODE_INFERENCE` (`complex_noise`, `random`, etc.).
- Genera partículas solo donde hay estado significativo (umbral dinámico: 0.01% del máximo).
- Optimizado para grids grandes (muestreo si `grid_size > 256`).

**Resultado:**
- Las partículas ahora emergen del estado inicial generado según la ley M.
- Consistencia completa con el motor Python.
- Respeta `INITIAL_STATE_MODE_INFERENCE`.

#### 2. Deprecación de `add_initial_particles()` ✅

**Cambios:**
- Método marcado como `DEPRECADO` con warning.
- Solo se mantiene como fallback temporal si la generación automática falla.
- Documentado claramente que es un hack temporal.

#### 3. Logging Mejorado para Diagnóstico ✅

**Archivos modificados:**
- `src/pipelines/core/websocket_handler.py`: Logging `INFO` para comandos recibidos, handlers encontrados, y completados.
- `src/pipelines/handlers/inference_handlers.py`: Logging al inicio de `handle_play()`.
- `src/pipelines/pipeline_server.py`: Logging al inicio de `handle_load_experiment()` y `handle_unload_model()`.

**Beneficios:**
- Diagnóstico más fácil de problemas de comunicación WebSocket.
- Visibilidad completa del flujo de comandos.
- Logging de handlers disponibles si comando es desconocido.

### Resultados
- ✅ Estado inicial generado correctamente según ley M.
- ✅ Partículas emergen del estado denso, no se agregan manualmente.
- ✅ Logging suficiente para diagnosticar problemas de comandos WebSocket.
- ⚠️ **Pendiente**: Verificar que los comandos WebSocket funcionen correctamente después de estos cambios.

### Archivos Modificados
- `src/engines/native_engine_wrapper.py` - Generación de estado inicial
- `src/pipelines/core/websocket_handler.py` - Logging mejorado
- `src/pipelines/handlers/inference_handlers.py` - Logging mejorado
- `src/pipelines/pipeline_server.py` - Logging mejorado

### Referencias
- [[00_KNOWLEDGE_BASE.md]] - Base de conocimientos del proyecto
- [[VISUALIZATION_FIX_ROADMAP.md]] - Roadmap de corrección de visualización

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
