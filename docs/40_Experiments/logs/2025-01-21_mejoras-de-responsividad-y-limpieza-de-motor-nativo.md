## 2025-01-21 - Mejoras de Responsividad y Limpieza de Motor Nativo

### Contexto
Se identificaron dos problemas críticos durante la inferencia:
1. **Comandos WebSocket tardaban en procesarse** - El `simulation_loop` bloqueaba el event loop
2. **Servidor se cerraba al limpiar motor nativo** - El método `cleanup()` podía causar errores no manejados

### Problemas Resueltos

#### 1. Responsividad de Comandos WebSocket

**Antes:**
- El `simulation_loop` ejecutaba muchos pasos sin yield al event loop
- Los comandos WebSocket tardaban en procesarse durante la inferencia
- Era necesario pausar y reanudar para que los comandos se ejecutaran

**Después:**
- ✅ Yield periódico al event loop durante ejecución de pasos
- ✅ Yield después de operaciones bloqueantes (conversión, visualización)
- ✅ Los comandos WebSocket se procesan inmediatamente

**Implementación:**

1. **Yield periódico en bucle de pasos** (`src/pipelines/core/simulation_loop.py`):
   - Cada 10 pasos para motor nativo (más frecuente por ser bloqueante)
   - Cada 50 pasos para motor Python
   - Permite procesar comandos WebSocket periódicamente

2. **Yield después de operaciones bloqueantes:**
   - Después de `get_dense_state()` (conversión puede tardar en grids grandes)
   - Después de `get_visualization_data()` (cálculo puede ser bloqueante)
   - Después de cada paso en modo live_feed

**Resultado:**
- Los comandos WebSocket ahora se procesan inmediatamente
- No es necesario pausar/reanudar para que los comandos se ejecuten
- La simulación sigue siendo rápida pero permite interrupciones frecuentes

#### 2. Limpieza Robusta de Motor Nativo

**Antes:**
- El servidor se cerraba cuando se limpiaba el motor nativo al cambiar experimentos
- El método `cleanup()` podía causar errores no manejados
- No había manejo de errores robusto alrededor de `cleanup()`

**Después:**
- ✅ Try-except específico alrededor de `cleanup()`
- ✅ Limpieza manual de respaldo si `cleanup()` falla
- ✅ Manejo de errores granular en cada paso de limpieza
- ✅ El servidor continúa funcionando incluso si hay errores durante la limpieza

**Implementación:**

1. **Manejo robusto en `pipeline_server.py`** (líneas 1014-1042):
   - Try-except específico alrededor de `old_motor.cleanup()`
   - Limpieza manual de respaldo si `cleanup()` falla
   - Captura de errores en cada paso individual

2. **Mejora en `NativeEngineWrapper.cleanup()`** (`src/engines/native_engine_wrapper.py`):
   - Manejo de errores granular para cada paso de limpieza
   - Continúa limpiando aunque un paso falle
   - Evita que errores críticos cierren el servidor

**Resultado:**
- El servidor ya no se cierra al cambiar entre experimentos
- La limpieza intenta múltiples estrategias antes de fallar
- Los errores se registran sin cerrar el servidor

#### 3. Corrección de Versión en setup.py

**Problema:**
- `setup.py` tenía `version="4.0.0"` cuando debería ser `4.1.1`
- Esto causaba que se instalara la versión incorrecta

**Solución:**
- Actualizado `setup.py` para usar `version="4.1.1"` desde `src/__version__.py`

### Archivos Modificados

1. **`src/pipelines/core/simulation_loop.py`**:
   - Yield periódico en bucle de pasos (líneas 117-120)
   - Yield después de `get_dense_state()` (líneas 263, 515)
   - Yield después de `get_visualization_data()` (líneas 340, 536)

2. **`src/pipelines/pipeline_server.py`**:
   - Manejo robusto de `cleanup()` del motor nativo (líneas 1014-1042)
   - Limpieza manual de respaldo si `cleanup()` falla

3. **`src/engines/native_engine_wrapper.py`**:
   - Manejo de errores granular en `cleanup()` (líneas 521-575)
   - Captura de errores individuales para cada paso de limpieza

4. **`setup.py`**:
   - Actualizado `version="4.1.1"` (línea 170)

5. **`.cursorrules`**:
   - Actualizado para que agentes revisen docs y hagan commits regularmente
   - Mejoras en documentación sobre commits y versionado

### Referencias
- `src/pipelines/core/simulation_loop.py` - Optimizaciones de yield
- `src/pipelines/pipeline_server.py` - Manejo robusto de cleanup
- `src/engines/native_engine_wrapper.py` - Cleanup granular

---


## [2025-11-23] Refactorización de Arquitectura: Servicios Desacoplados

### Contexto
La arquitectura anterior basada en un bucle monolítico (`simulation_loop.py`) presentaba problemas de bloqueo cuando operaciones pesadas (como `get_dense_state` o compresión) tardaban más de lo esperado, afectando la capacidad de respuesta del servidor a comandos (como "pausa").

### Cambios Realizados
1.  **Arquitectura de Servicios:** Se migró a una arquitectura basada en servicios orquestados por `ServiceManager`.
    -   `SimulationService`: Ejecuta el motor físico de forma aislada.
    -   `DataProcessingService`: Maneja la extracción de datos, visualización y compresión.
    -   `WebSocketService`: Gestiona la comunicación con clientes.
2.  **Desacoplamiento:** Uso de `asyncio.Queue` para comunicar servicios, permitiendo que la simulación continúe a su propio ritmo incluso si la visualización se retrasa (frame skipping).
3.  **Alineación de Visión:** Se eliminó la lógica de inyección artificial de partículas en `simulation_loop.py` para respetar el principio de "Emergencia" del proyecto. Las partículas deben surgir del vacío o ser sembradas explícitamente, no inyectadas como fallback.

### Impacto Esperado
-   **Mayor Responsividad:** Los comandos de control (pausa, stop) deberían procesarse inmediatamente.
-   **Mejor Rendimiento:** La simulación no debería verse ralentizada por la visualización.
-   **Modularidad:** Facilita la futura separación en microservicios o procesos distintos si fuera necesario.

### Referencias
- [[30_Components/SERVICE_ARCHITECTURE.md]] - Documentación de la nueva arquitectura


---
[[AI_DEV_LOG|🔙 Volver al Índice]]
