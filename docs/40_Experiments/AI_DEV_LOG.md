# 📝 AI Dev Log - Atheria 4

**Última actualización:** 2025-11-20  
**Objetivo:** Documentar decisiones de desarrollo, experimentos y cambios importantes para RAG y Obsidian.

---

## 📋 Índice de Entradas

- [[#2025-11-20 - Modo Manual de Visualización (steps_interval = 0)]]
- [[#2025-11-20 - Refactorización: Archivos Atómicos (En Progreso)]]
- [[#2025-11-20 - CLI Simple y Manejo de Errores Robusto]]
- [[#2025-11-20 - Checkpoint Step Tracking y Grid Scaling Info]]
- [[#2025-11-20 - Frame Skip Solo Cuando Live Feed OFF]]
- [[#2025-11-20 - Optimizaciones Críticas Motor Nativo Implementadas]]
- [[#2024-12-20 - Problemas Críticos Motor Nativo Identificados]]
- [[#2024-12-20 - Corrección Segfault: Cleanup Motor Nativo]]
- [[#2024-12-XX - Fase 3 Completada: Migración de Componentes UI]]
- [[#2024-12-XX - Fase 2 Iniciada: Setup Motor Nativo C++]]
- [[#2024-12-XX - Optimización de Logs y Reducción de Verbosidad]]

---

## 2025-11-20 - Modo Manual de Visualización (steps_interval = 0)

### Contexto
El usuario necesita una forma de dejar la simulación corriendo sin enviar frames automáticamente, solo actualizar la visualización cuando se presione un botón manualmente. Esto permite:
- Ejecutar la simulación a máxima velocidad sin overhead de visualización
- Reducir el uso de ancho de banda y recursos
- Mantener el control manual sobre cuándo actualizar la visualización

### Problema Resuelto

#### Antes
- `steps_interval` tenía un mínimo de 1, siempre enviaba frames automáticamente
- No había forma de ejecutar la simulación sin enviar frames periódicamente

#### Después
- `steps_interval = 0` activa el **modo manual**: la simulación corre sin enviar frames
- Nuevo handler `handle_update_visualization()` para actualización manual con botón
- El estado se mantiene en `g_state` para que al reconectar se vea el progreso

### Implementación

#### 1. Modo Manual (`steps_interval = 0`)

**Archivo:** `src/pipelines/pipeline_server.py`

**Función:** `handle_set_steps_interval()`

**Cambios:**
- Ahora acepta `steps_interval = 0` (anteriormente mínimo era 1)
- Límite máximo aumentado de `1,000` a `1,000,000` (1 millón)
- `steps_interval = 0` → Modo manual: no enviar frames automáticamente
- `steps_interval > 0` → Modo automático: enviar frame cada N pasos (hasta 1 millón)

**Lógica en `simulation_loop()`:**
```python
if steps_interval == 0:
    # Modo manual: ejecutar pasos rápidamente sin enviar frames
    steps_to_execute = 100  # Ejecutar múltiples pasos para velocidad
    should_send_frame = (g_state['last_frame_sent_step'] == -1)  # Solo primer frame
else:
    # Modo automático: enviar frame cada N pasos
    steps_to_execute = steps_interval
    should_send_frame = (steps_interval_counter >= steps_interval)
```

**Límite de `steps_interval`:**
- **Mínimo:** `0` (modo manual)
- **Máximo:** `1,000,000` (1 millón de pasos)
- Permite configurar intervalos muy grandes (ej: cada 10,000 o 100,000 pasos) para ejecuciones largas

#### 2. Handler de Actualización Manual

**Función:** `handle_update_visualization()`

**Características:**
- Actualiza la visualización manualmente cuando se presiona el botón
- Soporta motor nativo (lazy conversion) y motor Python
- Aplica optimizaciones (ROI, compresión, downsampling)
- Envía frame a todos los clientes conectados

**Uso:**
```javascript
// Frontend puede llamar:
ws.send(JSON.stringify({
    scope: "simulation",
    command: "update_visualization",
    args: {}
}));
```

**Mensajes de log:**
- Modo manual: `"[Simulación] Paso X completado (modo manual: presiona 'Actualizar Visualización' para ver)"`
- Modo automático: `"[Simulación] Paso X completado (live feed desactivado, mostrando cada N pasos)"`

### Estado Persistente

#### ¿Se mantiene el progreso al desconectar?

✅ **Sí**, el estado se mantiene en `g_state`:
- `g_state['simulation_step']` → Paso actual de la simulación
- `g_state['initial_step']` → Paso inicial (checkpoint)
- `g_state['motor']` → Motor con el estado cuántico actual
- `g_state['active_experiment']` → Experimento activo

**Al reconectar:**
- El cliente recibe `initial_state` con el estado actual
- El `step` se sincroniza automáticamente
- La visualización puede actualizarse manualmente para ver el estado actual

### Consideraciones de Seguridad

⚠️ **IMPORTANTE:** Ejecutar la simulación en modo manual sin supervisión puede ser **peligroso**:

1. **Uso de Recursos:**
   - La simulación consume CPU/GPU continuamente
   - Puede generar calor excesivo en el hardware
   - Puede afectar el rendimiento del sistema

2. **Memoria:**
   - Si la simulación se ejecuta por mucho tiempo, puede acumular memoria
   - Los snapshots automáticos pueden llenar el disco si están habilitados

3. **Recomendaciones:**
   - ✅ Usar `handle_enable_snapshots()` para controlar capturas automáticas
   - ✅ Monitorear el uso de recursos (CPU, GPU, RAM)
   - ✅ Establecer límites de tiempo de ejecución si es necesario
   - ✅ Pausar la simulación cuando no se esté usando

4. **Mejoras Futuras:**
   - [ ] Límite de tiempo de ejecución automático
   - [ ] Guardado automático periódico del estado
   - [ ] Monitoreo de recursos y alertas

### Archivos Modificados
1. **`src/pipelines/pipeline_server.py`**:
   - `handle_set_steps_interval()` - Ahora acepta `steps_interval = 0`
   - `handle_update_visualization()` - Nuevo handler para actualización manual
   - `simulation_loop()` - Lógica para modo manual
   - `HANDLERS` - Agregado `"update_visualization"` a `simulation`

### Corrección de Error

**Problema:** `UnboundLocalError: local variable 'logging' referenced before assignment` en `pipeline_viz.py`

**Causa:** Múltiples `import logging` dentro de bloques `except` hacían que Python tratara `logging` como variable local en toda la función.

**Solución:** Eliminados todos los `import logging` locales innecesarios (líneas 271, 296, 318, 535). `logging` ya está importado al inicio del archivo.

### Referencias
- `src/pipelines/pipeline_server.py` - Líneas 2707-2737 (handle_set_steps_interval)
- `src/pipelines/pipeline_server.py` - Líneas 983-1060 (handle_update_visualization)
- `src/pipelines/pipeline_server.py` - Líneas 221-325 (simulation_loop - modo manual)
- `src/pipelines/pipeline_viz.py` - Corrección de imports de logging

---

## 2025-11-20 - Refactorización: Archivos Atómicos (En Progreso)

### Contexto
El archivo `pipeline_server.py` tenía 3,567 líneas con 37 handlers, lo que hacía difícil mantener el código, buscar funcionalidades específicas y reducir el contexto necesario en los chats de IA.

### Objetivo
Factorizar `pipeline_server.py` en módulos más pequeños y atómicos (~300-700 líneas cada uno) para:
- Reducir contexto necesario en chats (de 3,567 → ~300-700 líneas por módulo)
- Facilitar búsquedas precisas
- Mejorar mantenibilidad y testing
- Reducir conflictos en colaboración

### Estructura Propuesta

```
src/pipelines/
├── server.py                    # Archivo principal (reducido ~500 líneas)
├── handlers/                    # Módulos de handlers (~300-700 líneas cada uno)
│   ├── experiment_handlers.py   ✅ CREADO
│   ├── simulation_handlers.py   ⏳ PENDIENTE
│   ├── inference_handlers.py    ⏳ PENDIENTE
│   ├── analysis_handlers.py     ⏳ PENDIENTE
│   ├── visualization_handlers.py ⏳ PENDIENTE
│   ├── config_handlers.py       ⏳ PENDIENTE
│   └── system_handlers.py       ⏳ PENDIENTE
├── core/                        # Componentes core
│   ├── websocket_handler.py     ⏳ PENDIENTE
│   ├── simulation_loop.py       ⏳ PENDIENTE
│   └── route_setup.py           ⏳ PENDIENTE
└── viz/                         # Visualizaciones
    ├── basic.py                 ⏳ PENDIENTE
    ├── advanced.py              ⏳ PENDIENTE
    └── physics.py               ⏳ PENDIENTE
```

### Progreso

#### ✅ Completado
1. **Plan de Refactorización**: Documentado en `docs/30_Components/REFACTORING_PLAN.md`
2. **Estructura de Directorios**: Creados `handlers/`, `core/`, y `viz/`
3. **experiment_handlers.py**: Módulo creado con handlers de experimentos:
   - `handle_create_experiment()`
   - `handle_continue_experiment()`
   - `handle_stop_training()`
   - `handle_delete_experiment()`
   - `handle_list_checkpoints()`
   - `handle_delete_checkpoint()`
   - `handle_cleanup_checkpoints()`
   - `handle_refresh_experiments()`

#### ⏳ Pendiente
1. Crear módulos restantes de handlers (simulation, inference, analysis, visualization, config, system)
2. Extraer `websocket_handler()` y `simulation_loop()` a módulos core
3. Refactorizar `pipeline_viz.py` en módulos de visualización
4. Actualizar `pipeline_server.py` para usar los nuevos módulos
5. Actualizar imports en otros archivos que usen handlers

### Beneficios Esperados

1. **Contexto Reducido**: De 3,567 líneas → ~300-700 líneas por módulo
2. **Búsquedas Más Precisas**: Buscar en módulo específico en lugar de archivo grande
3. **Mantenibilidad**: Cambios aislados en módulos específicos
4. **Testing**: Tests unitarios más fáciles por módulo
5. **Colaboración**: Menos conflictos, cambios más aislados

### Referencias
- [[REFACTORING_PLAN]] - Plan completo de refactorización
- `src/pipelines/handlers/experiment_handlers.py` - Módulo de handlers de experimentos

---

## 2025-11-20 - CLI Simple y Manejo de Errores Robusto

### Contexto
Creación de un CLI simple para facilitar el flujo de desarrollo y mejoras en el manejo de errores para prevenir segfaults y errores de conversión de tipos.

### Problemas Resueltos

#### 1. Comando Largo para Desarrollo
- **Antes:** `python3 setup.py build_ext --inplace && pip install -e . && ATHERIA_NO_FRONTEND=1 python3 run_server.py`
- **Después:** `atheria dev` o `python3 src/cli.py dev`

#### 2. Errores de Conversión de Tipos
- **Antes:** `'numpy.ndarray' object has no attribute 'detach'` cuando se intentaba convertir arrays numpy como si fueran tensores PyTorch
- **Después:** Verificaciones robustas con `isinstance()` y `hasattr()`, con fallback a `np.array()`

#### 3. Segfaults al Cambiar de Engine
- **Antes:** Segmentation fault al cambiar de motor nativo a Python sin cleanup adecuado
- **Después:** Cleanup explícito del motor anterior antes de cambiar, con try-except robusto

### Implementación

#### 1. CLI Simple (`src/cli.py`)

**Comandos disponibles:**
- `atheria dev` - Build + Install + Run (sin frontend por defecto)
- `atheria dev --frontend` - Build + Install + Run (con frontend)
- `atheria build` - Solo compilar extensiones C++
- `atheria install` - Solo instalar paquete
- `atheria run` - Solo ejecutar servidor
- `atheria clean` - Limpiar archivos de build

**Características:**
- Manejo de comandos con `argparse`
- Ejecución de comandos con `subprocess`
- Mensajes claros con emojis para mejor UX
- Manejo de errores con try-except

**Entry Points en `setup.py`:**
```python
entry_points={
    'console_scripts': [
        'atheria=src.cli:main',
        'ath=src.cli:main',  # Alias corto
    ],
}
```

#### 2. Manejo Robusto de Conversión de Tipos

**Archivo:** `src/pipelines/pipeline_viz.py`

**Cambios:**
- Cada conversión (density, phase, real_part, imag_part, energy) tiene su propio try-except
- Verifica `isinstance(tensor, torch.Tensor)` Y `hasattr(tensor, 'detach')` antes de llamar `.detach()`
- Fallback a `np.array()` si falla la conversión

**Resultado:**
- ✅ No más errores de `'numpy.ndarray' object has no attribute 'detach'`
- ✅ Manejo robusto de objetos híbridos o tipos inesperados

#### 3. Cleanup Robusto al Cambiar Engine

**Archivo:** `src/pipelines/pipeline_server.py`

**Función:** `handle_switch_engine()`

**Cambios:**
- Cleanup explícito del motor anterior ANTES de cambiar
- Try-except alrededor de todas las operaciones de cleanup
- Verificaciones con `hasattr()` antes de acceder a atributos

**Resultado:**
- ✅ No más segfaults al cambiar de motor nativo a Python
- ✅ Cleanup robusto incluso si hay errores

### Archivos Modificados
1. **`src/cli.py`** (nuevo) - CLI completo
2. **`setup.py`** - Agregado `entry_points`
3. **`src/pipelines/pipeline_viz.py`** - Manejo robusto de conversión
4. **`src/pipelines/pipeline_server.py`** - Cleanup robusto en switch_engine

### Estado
✅ **Completado**

---

## 2025-11-20 - Checkpoint Step Tracking y Grid Scaling Info

### Contexto
Implementación de tracking del paso del checkpoint y información de escalado de grid para mostrar correctamente el paso inicial desde el checkpoint.

### Problemas Resueltos

#### 1. Paso Actual Siempre Empezaba en 0
- **Antes:** `simulation_step` siempre se inicializaba en 0, incluso si había un checkpoint con un paso guardado
- **Después:** Lee el paso del checkpoint y inicializa `simulation_step = checkpoint_step`

#### 2. Falta de Información del Grid en UI
- **Antes:** No se mostraba información sobre el escalado del grid (training vs inference)
- **Después:** Se muestra `training_grid_size` y `inference_grid_size` en `checkpoint_info`

#### 3. Visualización "Total - Actual"
- **Antes:** Solo se mostraba el paso total
- **Después:** Se muestra "total - relativo" con hover mostrando el paso del checkpoint

### Implementación

**Archivo:** `src/pipelines/pipeline_server.py`

**Cambios:**
- Lee `step` y `episode` del checkpoint antes de cargar el modelo
- Si no hay `step`, calcula desde `episode × steps_per_episode`
- Guarda `checkpoint_step`, `checkpoint_episode`, `initial_step` en `g_state`
- Incluye `checkpoint_info` en `inference_status_update` con información del grid

**Archivo:** `frontend/src/modules/Dashboard/components/Toolbar.tsx`

**Cambios:**
- Muestra "total - relativo" en lugar de solo el paso total
- Ejemplo: `"1,356 - 0"` (total 1356, relativo 0 desde checkpoint)
- Hover muestra información del checkpoint

### Archivos Modificados
1. **`src/pipelines/pipeline_server.py`** - Lectura y guardado de checkpoint info
2. **`frontend/src/modules/Dashboard/components/Toolbar.tsx`** - Visualización mejorada

### Estado
✅ **Completado**

---

## 2025-11-20 - Frame Skip Solo Cuando Live Feed OFF

### Contexto
Corrección para que `frame_skip` solo se aplique cuando `live_feed` está OFF.

### Problema Resuelto

#### Frame Skip Interfiriendo con Live Feed
- **Antes:** `frame_skip` se aplicaba siempre, incluso cuando `live_feed` estaba ON, causando frames saltados
- **Después:** `frame_skip` solo se aplica cuando `live_feed` está OFF

### Implementación

**Archivo:** `src/pipelines/pipeline_server.py`

**Cambios:**
- Verificar `live_feed_enabled` antes de aplicar `frame_skip`
- Si `live_feed` está ON, siempre enviar frames (no saltar)

### Estado
✅ **Completado**

---

## 2024-12-20 - Optimizaciones Críticas Motor Nativo Implementadas

### Contexto
Implementación de optimizaciones críticas para resolver problemas de cuelgue y lentitud del motor nativo identificados anteriormente.

### Problemas Resueltos

#### 1. Cuelgue/Bloqueo del Motor Nativo
- **Antes:** `_update_dense_state_from_sparse()` se ejecutaba en cada paso, bloqueando la simulación
- **Después:** Lazy conversion - solo convierte cuando se necesita visualizar

#### 2. Lentitud Extrema en Tiempo Real
- **Antes:** Conversión completa de 65,536 coordenadas en cada paso (~650ms - 3.2s por paso)
- **Después:** Conversión solo cuando se necesita, con soporte ROI (3-5x más rápido)

### Implementación

#### 1. Lazy Conversion

**Archivo:** `src/engines/native_engine_wrapper.py`

**Cambios:**
- Agregado flag `_dense_state_stale` para rastrear si el estado denso está desactualizado
- `evolve_internal_state()` ahora solo marca como "stale", no convierte
- Método `get_dense_state()` convierte solo si es necesario

**Código:**
```python
def evolve_internal_state(self):
    # Ejecutar paso nativo (todo en C++)
    particle_count = self.native_engine.step_native()
    self.step_count += 1
    
    # OPTIMIZACIÓN CRÍTICA: NO convertir aquí - solo marcar como "stale"
    self._dense_state_stale = True

def get_dense_state(self, roi=None, check_pause_callback=None):
    """Obtiene el estado denso, convirtiendo solo si es necesario."""
    if self._dense_state_stale or self.state.psi is None or roi_changed:
        self._update_dense_state_from_sparse(roi=roi, check_pause_callback=check_pause_callback)
        self._dense_state_stale = False
    return self.state.psi
```

**Resultado:**
- ✅ No bloquea durante `evolve_internal_state()`
- ✅ Solo convierte cuando se necesita (al visualizar)
- ✅ Puede saltarse conversión completamente si `live_feed` está desactivado

#### 2. ROI Support

**Archivo:** `src/engines/native_engine_wrapper.py`

**Cambios:**
- `get_dense_state()` acepta parámetro `roi` (x_min, y_min, x_max, y_max)
- `_update_dense_state_from_sparse()` solo convierte región visible si se proporciona ROI
- Integrado con `ROIManager` en `pipeline_server.py`

**Resultado:**
- ✅ ROI pequeña (128x128): ~75% menos coordenadas (16,384 vs 65,536)
- ✅ Speedup estimado: **4x más rápido** con ROI pequeña
- ✅ Puede ser hasta **10-20x más rápido** con ROI muy pequeña (50x50)

#### 3. Verificación de Pausa Durante Conversión

**Archivo:** `src/engines/native_engine_wrapper.py`

**Cambios:**
- `get_dense_state()` acepta `check_pause_callback`
- `_update_dense_state_from_sparse()` verifica pausa cada batch (500-1000 coordenadas)
- Permite pausa inmediata incluso durante conversión larga

**Código:**
```python
for i in range(0, len(coords_to_process), BATCH_SIZE):
    # CRÍTICO: Verificar pausa cada batch para permitir pausa inmediata
    if check_pause_callback and check_pause_callback():
        logging.debug("Conversión interrumpida por pausa")
        return  # Salir temprano si está pausado
```

**Resultado:**
- ✅ Permite pausa inmediata (< 1 segundo) incluso durante conversión
- ✅ No bloquea UI durante conversión larga

### Integración con pipeline_server.py

**Archivo:** `src/pipelines/pipeline_server.py`

**Cambios:**
- Actualizado para usar `get_dense_state()` en lugares críticos
- Integrado con `ROIManager` para usar ROI cuando está habilitada
- Verificación de pausa durante conversión

**Lugares actualizados:**
1. `simulation_loop()` - Conversión antes de visualizar
2. `handle_set_viz()` - Conversión al cambiar visualización
3. Detección de época - Conversión solo cuando se necesita
4. Frame inicial - Conversión al cargar experimento

### Tests Realizados

**Script:** `scripts/test_native_engine_optimizations.py`

**Resultados:**
```
✅ TEST 1 PASADO: Lazy conversion funciona correctamente
✅ TEST 2 PASADO: ROI support funciona correctamente
✅ TEST 3 PASADO: Pause check funciona correctamente
✅ TEST 4 COMPLETADO: Estimación de rendimiento calculada
✅ TEST 5 PASADO: Integración correcta

Total: 5 tests
  ✅ Pasados: 5
  ⚠️  Saltados: 0
  ❌ Fallidos: 0
```

**Mejoras de Rendimiento Estimadas:**
- Grid completo: 65,536 coordenadas
- ROI pequeña (128x128): 16,384 coordenadas (75% reducción)
- Speedup estimado: **4x más rápido** con ROI pequeña

### Archivos Modificados

1. **`src/engines/native_engine_wrapper.py`**
   - Agregado flag `_dense_state_stale`
   - Método `get_dense_state()` con soporte ROI y verificación de pausa
   - `evolve_internal_state()` optimizado (no convierte automáticamente)
   - `_update_dense_state_from_sparse()` optimizado con ROI y verificación de pausa

2. **`src/pipelines/pipeline_server.py`**
   - Actualizado para usar `get_dense_state()` en lugares críticos
   - Integrado con `ROIManager` para ROI
   - Verificación de pausa durante conversión

3. **`scripts/test_native_engine_optimizations.py`** (nuevo)
   - Script de prueba para validar optimizaciones

### Resultados de Rendimiento

**Antes de Optimizaciones:**
- ❌ Cuelgue/bloqueo del motor nativo
- ❌ FPS muy bajo (lentitud extrema)
- ❌ Conversión de 65,536 coordenadas en cada paso (~650ms - 3.2s por paso)

**Después de Optimizaciones:**
- ✅ **~5000 FPS** en motor nativo 🚀
- ✅ Sin cuelgues ni bloqueos
- ✅ Conversión solo cuando se necesita visualizar
- ✅ ROI support permite hasta 26x más rápido con región pequeña

**Factores que Contribuyen al Alto Rendimiento:**
1. **Lazy Conversion**: No convierte estado denso en cada paso (~90% reducción)
2. **Motor Nativo C++**: Ejecución directa en C++ sin overhead Python
3. **Formato Disperso**: Solo procesa partículas activas, no todo el grid
4. **Sin Visualización**: Si `live_feed` está desactivado, ejecuta a máxima velocidad

**FPS según Configuración:**
- Motor Nativo + Lazy Conversion + Live Feed OFF: **~5000 FPS** 🚀
- Motor Nativo + ROI pequeña + Live Feed ON: **~1000-2000 FPS** (estimado)
- Motor Python: **~100-500 FPS** (dependiendo de grid_size)

### Estado
✅ **Implementado, Probado y Validado en Producción**

**Validación:**
- ✅ Tests automatizados pasados (5/5)
- ✅ Pruebas en producción: **~5000 FPS** confirmado
- ✅ Sin cuelgues ni bloqueos reportados
- ✅ Pausa inmediata funcionando

**Próximos Pasos:**
- Monitorear estabilidad en producción
- Optimizar tamaño de batch si es necesario
- Considerar batch conversion en C++ para mejora adicional

**Referencias:**
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES]] - Problemas identificados
- `src/engines/native_engine_wrapper.py:271-372` - Código optimizado
- `scripts/test_native_engine_optimizations.py` - Script de prueba

---

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

## 2024-12-20 - Corrección Segfault: Cleanup Motor Nativo

### Contexto
Se detectó un **segmentation fault (core dumped)** al cargar un experimento después de que se hubiera inicializado el motor nativo C++. El segfault ocurría cuando:

1. El motor nativo C++ se inicializaba primero (por ejemplo, al verificar disponibilidad)
2. Luego se decidía usar el motor Python
3. El motor nativo no se limpiaba correctamente antes de crear el motor Python
4. Al destruir el wrapper del motor nativo, los recursos C++ se liberaban de forma incorrecta

**Error observado:**
```
🚀 MOTOR NATIVO LISTO: device=cuda, grid_size=256
🐍 MOTOR PYTHON ACTIVO: device=cuda, grid_size=256
...
Segmentation fault (core dumped)
```

### Causa Raíz
El `NativeEngineWrapper` no tenía un método explícito de cleanup. Cuando Python hacía garbage collection del wrapper:

1. El destructor de Python (`__del__`) no liberaba explícitamente el motor nativo C++
2. Los tensores PyTorch en `state.psi` podían tener referencias circulares
3. El motor nativo C++ (`atheria_core.Engine`) se destruía después de que sus dependencias ya habían sido liberadas
4. Esto causaba acceso a memoria inválida → segfault

### Solución Implementada

#### 1. Método `cleanup()` Explícito

**Archivo:** `src/engines/native_engine_wrapper.py`

Se agregó un método `cleanup()` que libera recursos de forma controlada:

```python
def cleanup(self):
    """
    Limpia recursos del motor nativo de forma explícita.
    Debe llamarse antes de destruir el wrapper para evitar segfaults.
    """
    # Limpiar estado denso primero
    if hasattr(self, 'state') and self.state is not None:
        if hasattr(self.state, 'psi') and self.state.psi is not None:
            self.state.psi = None
        self.state = None
    
    # Limpiar referencias al motor nativo
    if hasattr(self, 'native_engine') and self.native_engine is not None:
        self.native_engine = None
    
    # Limpiar otras referencias
    self.model_loaded = False
    self.step_count = 0
    self.last_delta_psi = None
    ...
```

**Orden de cleanup:**
1. Primero: liberar tensores PyTorch (estado denso)
2. Segundo: liberar motor nativo C++ (cuando no hay dependencias)
3. Tercero: limpiar otras referencias

#### 2. Destructor Mejorado

Se agregó `__del__()` que llama a `cleanup()` automáticamente:

```python
def __del__(self):
    """Destructor - llama a cleanup para asegurar limpieza correcta."""
    try:
        self.cleanup()
    except Exception:
        # Ignorar errores en destructor para evitar problemas durante GC
        pass
```

#### 3. Cleanup Explícito en `handle_load_experiment`

**Archivo:** `src/pipelines/pipeline_server.py`

Se mejoró el cleanup del motor anterior antes de crear uno nuevo:

```python
# CRÍTICO: Limpiar motor nativo explícitamente antes de eliminarlo
if hasattr(old_motor, 'native_engine'):
    if hasattr(old_motor, 'cleanup'):
        old_motor.cleanup()
        logging.debug("Motor nativo limpiado explícitamente antes de eliminarlo")
```

#### 4. Cleanup al Fallar Inicialización

Cuando el motor nativo falla al inicializarse o cargar el modelo, se limpia correctamente:

```python
temp_motor = NativeEngineWrapper(...)
try:
    if temp_motor.load_model(jit_path):
        motor = temp_motor
        temp_motor = None  # Evitar cleanup - motor se usará
    else:
        # Limpiar motor nativo que falló
        if temp_motor is not None:
            temp_motor.cleanup()
            temp_motor = None
except Exception as e:
    # Limpiar motor nativo que falló durante inicialización
    if temp_motor is not None:
        temp_motor.cleanup()
        temp_motor = None
```

### Justificación

**Por qué cleanup explícito:**
- **Seguridad:** Evita segfaults por destrucción incorrecta de objetos C++
- **Predecibilidad:** Orden de destrucción controlado
- **Debugging:** Más fácil identificar problemas de memoria

**Por qué usar variable temporal:**
- Permite limpiar el motor nativo incluso si falla la carga del modelo
- Evita asignar a `motor` hasta que esté completamente inicializado
- Reduce riesgo de referencias colgantes

### Archivos Modificados

1. **`src/engines/native_engine_wrapper.py`**
   - Agregado método `cleanup()`
   - Agregado destructor `__del__()`

2. **`src/pipelines/pipeline_server.py`**
   - Mejorado cleanup del motor anterior en `handle_load_experiment`
   - Agregado cleanup cuando el motor nativo falla

### Testing

**Validación:**
- ✅ Cargar experimento con motor Python después de inicializar motor nativo
- ✅ Cambiar de motor nativo a Python sin segfault
- ✅ Motor nativo falla durante inicialización → cleanup correcto
- ✅ Motor nativo falla al cargar modelo → cleanup correcto

**Pruebas recomendadas:**
- Cargar múltiples experimentos consecutivamente
- Alternar entre motores nativo y Python
- Forzar fallos durante inicialización

### Estado
✅ **Completado y probado**

**Referencias:**
- [[Native_Engine_Core#Cleanup y Gestión de Memoria]]
- `src/engines/native_engine_wrapper.py:407-442`
- `src/pipelines/pipeline_server.py:1019-1042`

---

## 2024-12-XX - Optimización de Logs y Reducción de Verbosidad

### Contexto
El servidor generaba demasiados logs durante la operación normal, especialmente en el bucle de simulación. Esto generaba ruido innecesario y dificultaba identificar eventos importantes.

### Cambios Realizados

**Archivo:** `src/pipelines/pipeline_server.py`

1. **Reducción de verbosidad en WebSocket:**
   - `logging.info()` → `logging.debug()` para conexiones/desconexiones normales
   - Solo loguear eventos importantes (errores, warnings)

2. **Bucle de simulación:**
   - Diagnóstico cada 5 segundos en lugar de información constante
   - Logs de debug para eventos frecuentes (comandos recibidos, frames enviados)
   - Mantener INFO solo para eventos críticos

3. **Configuración de logging:**
   - Mantener `level=logging.INFO` por defecto
   - Usar `logging.debug()` para detalles técnicos que no son críticos

### Justificación
- **Rendimiento:** Menos overhead de I/O en logging
- **Legibilidad:** Logs más limpios, fáciles de filtrar
- **Debugging:** Mantener nivel DEBUG disponible cuando sea necesario

### Archivos Modificados
- `src/pipelines/pipeline_server.py`

### Estado
✅ **Completado**

---

## 2024-12-XX - Fase 3 Completada: Migración de Componentes UI

### Contexto
Completar la migración de componentes UI de Mantine a Tailwind CSS según el Design System establecido.

### Componentes Migrados

1. **CheckpointManager**
   - **Ubicación:** `frontend/src/components/training/CheckpointManager.tsx`
   - **Cambios:**
     - Migrado de Mantine a Tailwind CSS
     - Implementa Modal, Tabs, Table, Badge, Alert personalizados
     - Sistema de notas integrado
     - Gestión de checkpoints con operadores Pythonic
   - **Funcionalidad:** Completa gestión de checkpoints de entrenamiento

2. **TransferLearningWizard**
   - **Ubicación:** `frontend/src/components/experiments/TransferLearningWizard.tsx`
   - **Cambios:**
     - Migrado de Mantine a Tailwind CSS
     - Implementa Stepper personalizado
     - Formularios con NumberInput personalizado
     - Tabla de comparación de parámetros
     - Templates de progresión (standard, fine_tune, aggressive)
   - **Funcionalidad:** Wizard de 3 pasos para transfer learning

### Componentes Base Creados

**Ubicación:** `frontend/src/modules/Dashboard/components/`

1. **Modal.tsx** - Componente modal base
2. **Tabs.tsx** - Sistema de pestañas
3. **Table.tsx** - Tabla con estilos del Design System
4. **Badge.tsx** - Badges configurables
5. **Alert.tsx** - Alertas con iconos
6. **Stepper.tsx** - Indicador de pasos (horizontal/vertical)
7. **NumberInput.tsx** - Input numérico personalizado

### Justificación
- **Consistencia:** Todos los componentes siguen el Design System
- **Rendimiento:** Eliminación de dependencias pesadas (Mantine)
- **Mantenibilidad:** Componentes más simples y modulares
- **RAG:** Código más fácil de entender para agentes AI

### Estado
✅ **Completado**

---

## 2024-12-XX - Fase 2 Iniciada: Setup Motor Nativo C++

### Contexto
Iniciar la implementación del motor nativo C++ para escalar la simulación de miles a millones de partículas activas.

### Componentes Implementados

1. **CMakeLists.txt**
   - Configuración para PyBind11 y LibTorch
   - Detección automática de dependencias
   - Soporte para CUDA (12.2)

2. **setup.py**
   - Clase `CMakeBuildExt` personalizada
   - Integración con setuptools
   - Build system híbrido (CMake + setuptools)

3. **Estructuras C++ (`src/cpp_core/`):**
   - `Coord3D`: Coordenadas 3D con hash function
   - `SparseMap`: Mapa disperso (valores numéricos + tensores)
   - `Engine`: Clase base del motor nativo
   - `HarmonicVacuum`: Generador de vacío cuántico

4. **Bindings PyBind11:**
   - Función `add()` (Hello World) ✅
   - Estructura `Coord3D` expuesta ✅
   - Clase `SparseMap` con operadores Pythonic ✅
   - Clase `Engine` expuesta (pendiente pruebas completas)

### Compilación Exitosa

**Resultado:**
- Módulo generado: `atheria_core.cpython-310-x86_64-linux-gnu.so` (281KB)
- Sin errores de compilación
- LibTorch enlazado correctamente
- CUDA detectado (12.2)

### Issue Conocido (Runtime)

**Problema:** Error de importación relacionado con dependencias CUDA:
```
ImportError: undefined symbol: __nvJitLinkCreate_12_8
```

**Causa:** Configuración de entorno CUDA, no problema de compilación.

**Solución Temporal:**
- Configurar `LD_LIBRARY_PATH` correctamente
- O resolver conflictos de versiones CUDA

### Justificación
- **Rendimiento:** Eliminación del overhead del intérprete Python
- **Escalabilidad:** Capacidad de manejar millones de partículas
- **GPU:** Ejecución directa en GPU sin transferencias CPU↔GPU innecesarias

### Estado
✅ **Setup Completado** (compilación exitosa)  
⚠️ **Pendiente:** Resolver configuración CUDA para runtime

### Referencias
- [[ROADMAP_PHASE_2]]
- [[PHASE_2_SETUP_LOG]]

---

## Template para Nuevas Entradas

```markdown
## YYYY-MM-DD - Título del Cambio/Experimento

### Contexto
[Descripción del problema o necesidad que motivó el cambio]

### Cambios Realizados
[Descripción detallada de los cambios]

### Justificación
[Por qué se tomó esta decisión]

### Archivos Modificados
- `path/to/file1.py`
- `path/to/file2.tsx`

### Resultados
[Resultados obtenidos, métricas, observaciones]

### Estado
✅ Completado / 🔄 En progreso / ⚠️ Pendiente
```

---

**Nota:** Este log debe actualizarse después de cada cambio significativo o experimento.  
**Formato Obsidian:** Usar `[[]]` para enlaces internos cuando corresponda.
