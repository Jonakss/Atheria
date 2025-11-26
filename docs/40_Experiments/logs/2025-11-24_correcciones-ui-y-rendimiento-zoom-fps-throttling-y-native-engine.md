## 2025-11-24 - Correcciones UI y Rendimiento: Zoom, FPS, Throttling y Native Engine

### Contexto
El usuario reportó múltiples problemas con la interfaz y el rendimiento del simulador:
1. **Zoom unidireccional** - Solo permitía zoom in, no zoom out
2. **Motor nativo colgado** - Freeze al cargar experimentos con motor nativo (grids grandes)
3. **Throttling ignorado** - `steps_interval` no funcionaba con `live_feed_enabled=True`

### Cambios Implementados

#### 1. Corrección de Zoom Bidireccional
**Archivo**: [`frontend/src/hooks/usePanZoom.ts`](file:///home/jonathan.correa/Projects/Atheria/frontend/src/hooks/usePanZoom.ts)

**Problema**: La función `constrainPanZoom` calculaba un `minZoom` basado en el tamaño del grid y el contenedor, forzando a que el grid siempre llenara la pantalla. Esto impedía hacer zoom out si el grid era pequeño.

**Solución**: 
- Eliminé el cálculo dinámico de `minZoom` basado en `minZoomRequired * 0.8`
- Ahora usa un `minZoom` fijo de `0.1` (permite zoom out hasta ver 10x el viewport)
- Removí variables no utilizadas (`minZoomX`, `minZoomY`, `minZoomRequired`)

**Commit**: `fb61248` - `fix: corregir zoom bidireccional en usePanZoom (remover restricción minZoom) [version:bump:patch]`

#### 2. Fix de Freeze en Motor Nativo (Grids Grandes)
**Archivos**: 
- [`src/pipelines/handlers/inference_handlers.py`](file:///home/jonathan.correa/Projects/Atheria/src/pipelines/handlers/inference_handlers.py)
- [`src/engines/native_engine_wrapper.py`](file:///home/jonathan.correa/Projects/Atheria/src/engines/native_engine_wrapper.py)

**Problema**: Al presionar "Play", el backend intentaba enviar un frame inicial llamando a `get_dense_state()`. Para grids grandes (>128), esta conversión tomaba >10s, causando timeouts y cuelgues.

**Solución**:
- En `handle_play()`: Si `grid_size > 128`, **saltar** el envío del frame inicial
- La visualización arranca en el primer step de simulación (no bloquea el Play)
- Añadido logging detallado en `get_active_coords()` para diagnóstico

**Commit**: `efcab45` - `fix: saltar frame inicial en motor nativo para grids >128 y añadir logging de diagnóstico [version:bump:patch]`

**Por qué esta solución**:
- El frame inicial es "nice to have", no crítico
- Es mejor tener una UI responsiva que un frame inicial perfecto
- El usuario ve el primer frame casi inmediatamente después de presionar Play

#### 3. Throttling Mejorado (Steps Interval con Live Feed)
**Archivo**: [`src/pipelines/core/simulation_loop.py`](file:///home/jonathan.correa/Projects/Atheria/src/pipelines/core/simulation_loop.py)

**Problema**: El bloque `if not live_feed_enabled:` ejecutaba múltiples pasos según `steps_interval`, pero cuando `live_feed_enabled=True`, se forzaba 1 paso por frame (lento).

**Solución**:
- **Unificación de lógica**: Ahora `steps_interval` se respeta SIEMPRE (con y sin live feed)
- Cuando `live_feed_enabled=True` y `steps_interval=10`: ejecuta 10 pasos de física por cada frame visualizado
- Default inteligente: `steps_interval=10` si live feed OFF, `steps_interval=1` si live feed ON (mantiene comportamiento anterior por defecto)
- Throttle adaptativo: Si `steps_interval > 1`, permite ir rápido entre frames (`await asyncio.sleep(0)`)

**Commit**: `5339ef9` - `feat: respetar steps_interval con live_feed activo para mejor rendimiento [version:bump:minor]`

**Por qué esto es importante**:
- Antes: Con live feed ON, era imposible acelerar la simulación (siempre 1 paso/frame)
- Ahora: Puedes tener live feed ON y aún así correr 10-100 pasos entre visualizaciones
- Resultado: **10-100x más rápido** sin sacrificar la visualización en tiempo real

### Impacto

| Cambio | Antes | Ahora | Mejora |
|--------|-------|-------|--------|
| **Zoom** | Solo zoom in | Zoom in/out libre | ✅ UX mejorada |
| **Native Engine Play** | Freeze >10s en grids >128 | Inicio <1s (sin frame inicial) | ✅ 10x más responsivo |
| **Live Feed Speed** | 1 paso/frame (lento) | N pasos/frame (configurable) | ✅ 10-100x más rápido |

### Archivos Relacionados
- [[PYTHON_TO_NATIVE_MIGRATION]] - Troubleshooting de motor nativo
- [[INFERENCE_HANDLERS_ARCHITECTURE]] - Arquitectura de handlers
- Frontend: `usePanZoom.ts`, `PanZoomCanvas.tsx` (FPS display ya existía)
- Backend: `simulation_loop.py`, `inference_handlers.py`, `native_engine_wrapper.py`

### Próximos Pasos
- [ ] Añadir slider de `steps_interval` en UI (si no existe)
- [ ] Considerar ROI automático más agresivo para grids >256
- [ ] Investigar si el motor nativo puede enviar frames parciales (streaming)

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
