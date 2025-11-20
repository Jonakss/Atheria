# Plan de Refactorización - Archivos Atómicos

**Fecha:** 2025-11-20  
**Objetivo:** Factorizar archivos grandes en módulos más atómicos para facilitar búsquedas, reducir contexto en chats y mejorar mantenibilidad.

---

## 📊 Análisis de Archivos Grandes

### Archivos que Necesitan Refactorización

1. **`src/pipelines/pipeline_server.py`** - 3,567 líneas
   - 37 handlers async
   - Sistema de routing WebSocket
   - Loop de simulación
   - Configuración de rutas HTTP

2. **`src/server/server_handlers.py`** - 1,381 líneas
   - Handlers de entrenamiento
   - Lógica de creación de experimentos

3. **`src/pipelines/pipeline_viz.py`** - 543 líneas
   - Una función grande `get_visualization_data()`
   - Múltiples tipos de visualización

4. **`src/engines/native_engine_wrapper.py`** - 516 líneas
   - Wrapper del motor nativo
   - Conversión sparse ↔ dense
   - Lazy conversion y ROI

---

## 🎯 Estructura Propuesta

### 1. Refactorización de `pipeline_server.py`

**Estructura propuesta:**
```
src/pipelines/
├── __init__.py
├── server.py                    # Archivo principal (reducido ~500 líneas)
├── handlers/                    # Módulo de handlers
│   ├── __init__.py
│   ├── experiment_handlers.py   # ~400 líneas
│   ├── simulation_handlers.py   # ~600 líneas
│   ├── inference_handlers.py    # ~700 líneas
│   ├── analysis_handlers.py     # ~400 líneas
│   ├── visualization_handlers.py # ~300 líneas
│   ├── config_handlers.py       # ~300 líneas
│   └── system_handlers.py       # ~100 líneas
├── core/
│   ├── __init__.py
│   ├── websocket_handler.py     # ~150 líneas
│   ├── simulation_loop.py       # ~600 líneas
│   └── route_setup.py           # ~200 líneas
└── viz/
    ├── __init__.py
    ├── basic.py                 # density, phase, energy
    ├── advanced.py              # poincare, flow, attractors
    └── physics.py               # physics map, entropy
```

**Categorización de handlers:**

#### `handlers/experiment_handlers.py` (~400 líneas)
- `handle_create_experiment()`
- `handle_continue_experiment()`
- `handle_stop_training()`
- `handle_delete_experiment()`
- `handle_refresh_experiments()`

#### `handlers/simulation_handlers.py` (~600 líneas)
- `handle_play()`
- `handle_pause()`
- `handle_reset()`
- `handle_load_experiment()`
- `handle_switch_engine()`
- `handle_unload_model()`

#### `handlers/inference_handlers.py` (~700 líneas)
- `handle_set_viz()`
- `handle_set_simulation_speed()`
- `handle_set_fps()`
- `handle_set_frame_skip()`
- `handle_set_live_feed()`
- `handle_set_steps_interval()`
- `handle_inject_energy()`
- `handle_set_inference_config()`

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

#### `handlers/system_handlers.py` (~100 líneas)
- `handle_list_checkpoints()`
- `handle_cleanup_checkpoints()`
- `handle_delete_checkpoint()`

---

### 2. Refactorización de `pipeline_viz.py`

**Estructura propuesta:**
```
src/pipelines/viz/
├── __init__.py                  # Re-exporta funciones principales
├── base.py                      # Conversión de tipos, validación
├── basic.py                     # Visualizaciones básicas
│   - density()
│   - phase()
│   - energy()
│   - real_imaginary()
├── advanced.py                  # Visualizaciones avanzadas
│   - poincare()
│   - flow()
│   - phase_attractor()
│   - poincare_3d()
└── physics.py                   # Visualizaciones de física
    - physics_map()
    - entropy()
    - coherence()
```

**Archivo principal:**
```python
# src/pipelines/viz/__init__.py
from .basic import get_density, get_phase, get_energy
from .advanced import get_poincare, get_flow, get_phase_attractor
from .physics import get_physics_map, get_entropy
from .base import get_visualization_data  # Orquestador principal
```

---

### 3. Separación de `simulation_loop.py`

**Archivo:** `src/pipelines/core/simulation_loop.py`

**Contenido:**
- Función `simulation_loop()` principal
- Lógica de throttling y FPS
- Lógica de frame skipping
- Integración con lazy conversion y ROI

---

### 4. Separación de `websocket_handler.py`

**Archivo:** `src/pipelines/core/websocket_handler.py`

**Contenido:**
- Función `websocket_handler()`
- Manejo de mensajes WebSocket
- Routing de comandos
- Estado inicial del cliente

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

## 🎯 Prioridad

1. **Alta:** `pipeline_server.py` → Separar handlers (impacto inmediato en contexto)
2. **Media:** `pipeline_viz.py` → Separar visualizaciones (mejora búsquedas)
3. **Baja:** `native_engine_wrapper.py` → Ya está relativamente bien organizado

---

## 📚 Relacionado

- [[AI_DEV_LOG#2025-11-20 - CLI Simple y Manejo de Errores Robusto]]
- [[TECHNICAL_ARCHITECTURE_V4]]

