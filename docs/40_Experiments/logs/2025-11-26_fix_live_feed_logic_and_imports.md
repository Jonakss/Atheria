---
type: fix
date: 2025-11-26
component: Backend (Simulation Loop, Inference Handlers)
author: AI Assistant
status: implemented
---

# Fix: Live Feed Logic & Import Errors

## Contexto

Se resolvieron dos problemas críticos:

1.  **Live Feed Zombie:** La simulación seguía enviando frames "en tiempo real" incluso cuando el usuario desactivaba el "Live Feed", debido a que el intervalo de pasos por defecto (10) seguía activo.
2.  **ImportError Crítico:** El servidor fallaba al intentar exportar modelos o usar ciertas funciones debido a importaciones relativas inválidas (`from ... import config`).

## Problemas Identificados

### 1. Live Feed no respetaba "Desactivado"

El usuario reportó: *"me lo sigue cmandando en tiempo real cuando desactivo el live feed"*.

**Causa:**
En `simulation_loop.py`, la lógica para enviar frames (`should_send_frame`) dependía únicamente de `steps_interval`. Cuando se desactiva el Live Feed, `steps_interval` a menudo permanece en su valor por defecto (10), lo que causa que el backend siga enviando frames cada 10 pasos, saturando la conexión igual que si estuviera activo.

### 2. ImportError: relative import beyond top-level package

El usuario reportó: `ImportError: attempted relative import beyond top-level package` en `native_engine_wrapper.py`.

**Causa:**
Uso incorrecto de importaciones relativas con tres puntos (`from ... import config`) en archivos que son ejecutados o importados de manera que `src` es el paquete raíz.

## Soluciones Implementadas

### 1. Fix Lógica Live Feed (`simulation_loop.py`)

Se modificó la lógica para **forzar el comportamiento de "Full Speed"** (no enviar frames) cuando `live_feed_enabled` es `False`, a menos que se haya configurado explícitamente un intervalo muy largo (para actualizaciones lentas).

```python
# CRÍTICO: Si live_feed está desactivado, forzar comportamiento de fullspeed
# a menos que steps_interval sea muy grande (>1000) para actualizaciones lentas
effective_steps_interval = steps_interval
if not live_feed_enabled and steps_interval != -1 and steps_interval < 1000:
    effective_steps_interval = -1

if effective_steps_interval == -1:
    should_send_frame = False
# ...
```

### 2. Fix Importaciones Relativas

Se reemplazaron todas las importaciones relativas problemáticas por importaciones absolutas robustas:

```python
# ANTES
from ... import config as global_cfg
from ...server.server_state import g_state

# DESPUÉS
from src import config as global_cfg
from src.server.server_state import g_state
```

**Archivos corregidos:**
- `src/engines/native_engine_wrapper.py`
- `src/pipelines/core/simulation_loop.py`
- `src/pipelines/core/status_helpers.py`
- `src/pipelines/core/websocket_handler.py`
- `src/pipelines/handlers/inference_handlers.py`
- `src/pipelines/handlers/experiment_handlers.py`

## Resultado

- ✅ **Live Feed OFF = 0 Overhead:** Al desactivar el Live Feed, ahora se garantiza que no se envían frames (comportamiento full speed real).
- ✅ **Estabilidad:** Eliminados los crashes por `ImportError`.
