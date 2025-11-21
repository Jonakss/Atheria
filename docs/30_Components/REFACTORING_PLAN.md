# Plan de Refactorización - Archivos Atómicos

**Fecha:** 2025-11-20  
**Última actualización:** 2025-01-21  
**Objetivo:** Factorizar archivos grandes en módulos más atómicos para facilitar búsquedas, reducir contexto en chats y mejorar mantenibilidad.  
**Estado:** En progreso - Refactorización parcial completada (handlers básicos, visualizaciones, core modules)

---

## 📊 Análisis de Archivos Grandes

### Estado de Refactorización

#### ✅ Completado

1. **`src/pipelines/pipeline_viz.py`** ✅
   - **Antes:** 543 líneas monolíticas
   - **Después:** Paquete modular `src/pipelines/viz/`
     - `utils.py` - Utilidades (conversión, normalización)
     - `core.py` - Cálculos básicos y función principal
     - `advanced.py` - Visualizaciones avanzadas
   - **Estado:** Completado, wrapper de compatibilidad mantenido

2. **`src/pipelines/core/`** ✅
   - `simulation_loop.py` - Loop de simulación extraído (~700 líneas)
   - `websocket_handler.py` - Handler WebSocket extraído (~150 líneas)
   - `helpers.py` - Helpers adaptativos (downsample, ROI)
   - `status_helpers.py` - Helpers para status payloads
   - **Estado:** Completado

3. **`src/pipelines/handlers/`** 🔄 (Parcial)
   - `inference_handlers.py` - Handlers básicos (play, pause) ✅
   - `simulation_handlers.py` - Handlers de simulación (viz, speed, fps, live_feed) ✅
   - `system_handlers.py` - Handlers del sistema (shutdown, refresh) ✅
   - `experiment_handlers.py` - Ya existía, completado ✅
   - **Pendiente:** Handlers complejos aún en `pipeline_server.py`

#### 🔄 En Progreso

1. **`src/pipelines/pipeline_server.py`** 🔄
   - **Antes:** ~4,023 líneas
   - **Después:** ~4,000 líneas (handlers básicos extraídos)
   - **Pendiente:** 
     - Eliminar definiciones duplicadas de handlers ya extraídos
     - Extraer handlers complejos restantes:
       - `handle_load_experiment()` (~650 líneas)
       - `handle_switch_engine()` (~150 líneas)
       - `handle_unload_model()` (~150 líneas)
       - `handle_reset()` (~100 líneas)
       - `handle_inject_energy()` (~160 líneas)
       - Handlers de análisis (`analyze_universe_atlas`, `analyze_cell_chemistry`)
       - Handlers de configuración (snapshots, history, etc.)

#### ⚠️ Pendiente

1. **`src/server/server_handlers.py`** - 1,381 líneas
   - Handlers de entrenamiento
   - Lógica de creación de experimentos
   - **Prioridad:** Media

2. **`src/engines/native_engine_wrapper.py`** - 516 líneas
   - Wrapper del motor nativo
   - Conversión sparse ↔ dense
   - Lazy conversion y ROI
   - **Prioridad:** Baja (ya está relativamente bien organizado)

---

## 🎯 Estructura Actual

### 1. Refactorización de `pipeline_server.py`

**Estructura actual (implementada):**
```
src/pipelines/
├── __init__.py
├── pipeline_server.py           # Archivo principal (~4,000 líneas) 🔄
├── pipeline_viz.py              # Wrapper de compatibilidad ✅
├── handlers/                    # Módulo de handlers 🔄
│   ├── __init__.py              # ✅
│   ├── experiment_handlers.py   # ✅ (~310 líneas)
│   ├── inference_handlers.py    # ✅ (play, pause - ~55 líneas)
│   ├── simulation_handlers.py   # ✅ (viz, speed, fps, live_feed - ~235 líneas)
│   ├── system_handlers.py       # ✅ (shutdown, refresh - ~73 líneas)
│   ├── analysis_handlers.py     # ⚠️ Pendiente
│   ├── visualization_handlers.py # ⚠️ Pendiente
│   └── config_handlers.py       # ⚠️ Pendiente
├── core/                        # Módulos core ✅
│   ├── __init__.py              # ✅
│   ├── websocket_handler.py     # ✅ (~150 líneas)
│   ├── simulation_loop.py       # ✅ (~700 líneas)
│   ├── helpers.py               # ✅ (downsample, ROI)
│   └── status_helpers.py        # ✅ (status payloads)
└── viz/                         # Paquete de visualizaciones ✅
    ├── __init__.py              # ✅
    ├── utils.py                 # ✅ (utilidades, normalización)
    ├── core.py                  # ✅ (cálculos básicos)
    └── advanced.py              # ✅ (visualizaciones avanzadas)
```

**Categorización de handlers:**

#### `handlers/experiment_handlers.py` (~400 líneas)
- `handle_create_experiment()`
- `handle_continue_experiment()`
- `handle_stop_training()`
- `handle_delete_experiment()`
- `handle_refresh_experiments()`

#### `handlers/inference_handlers.py` ✅ (~55 líneas) - **Actual**
- `handle_play()` - Inicia simulación
- `handle_pause()` - Pausa simulación

**Pendiente en `pipeline_server.py`:**
- `handle_load_experiment()` (~650 líneas) - Carga experimento
- `handle_switch_engine()` (~150 líneas) - Cambia motor
- `handle_unload_model()` (~150 líneas) - Descarga modelo
- `handle_reset()` (~100 líneas) - Reinicia simulación
- `handle_inject_energy()` (~160 líneas) - Inyecta energía
- `handle_set_inference_config()` - Configuración de inferencia

#### `handlers/simulation_handlers.py` ✅ (~235 líneas) - **Actual**
- `handle_set_viz()` - Cambia visualización
- `handle_update_visualization()` - Actualización manual
- `handle_set_simulation_speed()` - Velocidad de simulación
- `handle_set_fps()` - FPS objetivo
- `handle_set_frame_skip()` - Frame skip
- `handle_set_live_feed()` - Live feed on/off
- `handle_set_steps_interval()` - Intervalo de pasos

**Pendiente en `pipeline_server.py`:**
- `handle_set_compression()` - Compresión de datos
- `handle_set_downsample()` - Downsampling
- `handle_set_roi()` - Region of Interest
- Handlers de snapshots y history

#### `handlers/analysis_handlers.py` (~400 líneas)
- `handle_analyze_universe_atlas()`
- `handle_analyze_cell_chemistry()`
- `handle_cancel_analysis()`

#### `handlers/visualization_handlers.py` (~300 líneas)
- `handle_set_compression()`
- `handle_set_downsample()`
- `handle_set_roi()`

#### `handlers/config_handlers.py` (~300 líneas)
- `handle_set_snapshot_interval()`
- `handle_enable_snapshots()`
- `handle_capture_snapshot()`
- `handle_clear_snapshots()`
- `handle_enable_history()`
- `handle_save_history()`
- `handle_clear_history()`
- `handle_list_history_files()`
- `handle_load_history_file()`

#### `handlers/system_handlers.py` ✅ (~73 líneas) - **Actual**
- `handle_shutdown()` - Apaga servidor desde UI
- `handle_refresh_experiments()` - Actualiza lista de experimentos

**Pendiente en `experiment_handlers.py`:**
- Ya incluidos: `handle_list_checkpoints()`, `handle_cleanup_checkpoints()`, `handle_delete_checkpoint()`

---

### 2. Refactorización de `pipeline_viz.py` ✅ **COMPLETADO**

**Estructura implementada:**
```
src/pipelines/viz/
├── __init__.py                  # ✅ Re-exporta get_visualization_data
├── utils.py                     # ✅ Conversión de tipos, normalización, helpers
│   - tensor_to_numpy()
│   - normalize_map_data()
│   - apply_downsampling()
│   - synchronize_gpu()
│   - get_inference_context()
├── core.py                      # ✅ Cálculos básicos y función principal
│   - get_visualization_data()   # Función orquestadora
│   - calculate_basic_quantities()
│   - calculate_gradient_magnitude()
│   - select_map_data()
│   - calculate_entropy_map()
│   - calculate_coherence_map()
│   - calculate_channel_activity_map()
│   - calculate_histograms()
└── advanced.py                  # ✅ Visualizaciones avanzadas
    - calculate_poincare_coords()
    - calculate_phase_attractor()
    - calculate_flow_data()
    - calculate_complex_3d_data()
    - calculate_phase_hsv_data()
```

**Archivo wrapper:**
```python
# src/pipelines/pipeline_viz.py - Mantiene compatibilidad
from .viz import get_visualization_data
__all__ = ['get_visualization_data']
```

**Beneficios obtenidos:**
- Separación clara de responsabilidades
- Código más mantenible y extensible
- Mejor organización para RAG

---

### 3. Separación de `simulation_loop.py` ✅ **COMPLETADO**

**Archivo:** `src/pipelines/core/simulation_loop.py` (~700 líneas)

**Contenido extraído:**
- ✅ Función `simulation_loop()` principal
- ✅ Lógica de throttling y FPS
- ✅ Lógica de frame skipping
- ✅ Integración con lazy conversion y ROI
- ✅ Adaptive downsampling y ROI automático
- ✅ Frame payload optimization
- ✅ History saving y snapshot capturing

**Estado:** Completado, funcionando correctamente

---

### 4. Separación de `websocket_handler.py` ✅ **COMPLETADO**

**Archivo:** `src/pipelines/core/websocket_handler.py` (~150 líneas)

**Contenido extraído:**
- ✅ Función `websocket_handler()`
- ✅ Manejo de mensajes WebSocket
- ✅ Estado inicial del cliente (incluye versiones de motores)
- ✅ Manejo robusto de errores (ConnectionResetError, ConnectionError, OSError, RuntimeError)

**Mejoras implementadas:**
- ✅ Mejor manejo de desconexiones
- ✅ Logging más informativo
- ✅ Factory pattern para crear handler con HANDLERS dictionary

**Estado:** Completado, funcionando correctamente

---

### 5. Helpers Extraídos ✅ **COMPLETADO**

**Archivo:** `src/pipelines/core/helpers.py`

**Funciones:**
- ✅ `calculate_adaptive_downsample()` - Downsampling adaptativo para grids grandes
- ✅ `calculate_adaptive_roi()` - ROI automático para grids grandes

**Estado:** Completado, en uso en `simulation_loop.py`

---

### 6. Status Helpers ✅ **COMPLETADO**

**Archivo:** `src/pipelines/core/status_helpers.py`

**Funciones:**
- ✅ `get_compile_status()` - Obtiene compile_status de g_state o lo reconstruye del motor activo
- ✅ `build_inference_status_payload()` - Construye payload de status con compile_status siempre incluido

**Beneficio:** Consistencia en todos los status updates, compile_status siempre presente

**Estado:** Completado, en uso en múltiples handlers

---

## 🔄 Estrategia de Migración

### Fase 1: Crear Estructura (Sin Romper Nada)
1. Crear nuevos directorios y archivos
2. Mover handlers a módulos separados
3. Mantener imports en `pipeline_server.py` temporalmente

### Fase 2: Actualizar Imports
1. Actualizar `__init__.py` para re-exportar
2. Actualizar imports en `pipeline_server.py`
3. Actualizar imports en otros archivos que usen handlers

### Fase 3: Limpiar
1. Eliminar código duplicado
2. Actualizar documentación
3. Verificar tests

---

## ✅ Beneficios Esperados

### 1. Contexto Reducido en Chats
- **Antes:** Archivo de 3,567 líneas → contexto completo necesario
- **Después:** Archivo de ~300 líneas → contexto específico

### 2. Búsquedas Más Precisas
- **Antes:** Buscar en 3,567 líneas
- **Después:** Buscar en archivo específico (~300-700 líneas)

### 3. Mantenibilidad
- **Antes:** Cambios en un handler afectan todo el archivo
- **Después:** Cambios aislados en módulo específico

### 4. Testing
- **Antes:** Tests requieren mockear todo el pipeline_server
- **Después:** Tests unitarios más fáciles por módulo

### 5. Colaboración
- **Antes:** Conflictos frecuentes en archivo grande
- **Después:** Menos conflictos, cambios más aislados

---

## 📝 Ejemplo de Estructura Final

### `src/pipelines/handlers/inference_handlers.py`
```python
"""Handlers para control de inferencia y simulación."""
from ...server.server_state import g_state, send_notification
from ...engines.qca_engine import QuantumState

async def handle_play(args):
    """Inicia la simulación."""
    # ... código específico ...
    
async def handle_pause(args):
    """Pausa la simulación."""
    # ... código específico ...
    
# ... otros handlers ...
```

### `src/pipelines/server.py` (archivo principal reducido)
```python
"""Servidor principal de Atheria."""
from .core.websocket_handler import websocket_handler
from .core.simulation_loop import simulation_loop
from .core.route_setup import setup_routes
from .handlers import (
    experiment_handlers,
    simulation_handlers,
    inference_handlers,
    # ... otros módulos ...
)

# Diccionario de handlers
HANDLERS = {
    "experiment": experiment_handlers.HANDLERS,
    "simulation": simulation_handlers.HANDLERS,
    "inference": inference_handlers.HANDLERS,
    # ...
}

# Resto del código de configuración del servidor
```

---

## ✅ Estado de Refactorización

### Completado
1. ✅ **`pipeline_viz.py`** → Paquete modular `viz/` (completado)
2. ✅ **`simulation_loop`** → `core/simulation_loop.py` (completado)
3. ✅ **`websocket_handler`** → `core/websocket_handler.py` (completado)
4. ✅ **Helpers** → `core/helpers.py` (completado)
5. ✅ **Status Helpers** → `core/status_helpers.py` (completado)
6. ✅ **Handlers básicos** → `handlers/inference_handlers.py`, `simulation_handlers.py`, `system_handlers.py` (completado)

### En Progreso
1. 🔄 **`pipeline_server.py`** → Eliminar duplicados de handlers extraídos
2. 🔄 **Handlers complejos** → Extraer cuando sea necesario o beneficioso

### Pendiente (Prioridad Media/Baja)
1. ⚠️ **Handlers de análisis** → `handlers/analysis_handlers.py` (universe_atlas, cell_chemistry)
2. ⚠️ **Handlers de configuración** → `handlers/config_handlers.py` (snapshots, history, etc.)
3. ⚠️ **`server_handlers.py`** → Ya está en directorio separado, menos crítico
4. ⚠️ **`native_engine_wrapper.py`** → Ya está relativamente bien organizado

## 🎯 Beneficios Obtenidos

### Contexto Reducido
- **Antes:** `pipeline_server.py` 4,023 líneas → contexto completo necesario
- **Después:** Módulos de ~50-700 líneas → contexto específico

### Mantenibilidad
- ✅ Código más modular y organizado
- ✅ Cambios aislados por módulo
- ✅ Más fácil de entender y modificar

### Testing
- ✅ Módulos testables independientemente
- ✅ Menos mocking necesario

### Búsquedas
- ✅ Búsquedas más precisas en archivos más pequeños
- ✅ Mejor organización para RAG

---

## 📚 Relacionado

- [[AI_DEV_LOG#2025-11-20 - CLI Simple y Manejo de Errores Robusto]]
- [[TECHNICAL_ARCHITECTURE_V4]]

