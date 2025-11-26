---
type: fix
date: 2025-11-26
component: Backend (Simulation Loop, Inference Handlers)
author: AI Assistant
status: implemented
---

# Fix: Native Engine Freeze & Grid Scaling

## Contexto

El usuario reportó dos problemas críticos:

1.  **Grid Scaling:** Al intentar establecer el tamaño del grid a 32 (tamaño original de entrenamiento), el sistema lo forzaba a 256 (tamaño por defecto de inferencia). El comando para cambiar la configuración no recargaba el motor, por lo que el cambio no surtía efecto.
2.  **Native Engine Freeze:** El motor nativo (C++) se bloqueaba ("se traca") al iniciar la simulación, causando que el servidor dejara de responder. Esto ocurría específicamente en la llamada a `step_native()`.

## Problemas Detectados

1.  **Inference Handlers:** `handle_set_inference_config` actualizaba `g_state` y `global_cfg` pero no reinicializaba el motor (`NativeEngineWrapper` o `Aetheria_Motor`). Como el tamaño del grid se define al construir el motor, el cambio no se aplicaba hasta un reinicio manual o recarga del experimento.
2.  **Simulation Loop:** La llamada a `motor.evolve_internal_state` (que llama a C++) se hacía de forma síncrona dentro de un `run_in_executor` pero sin timeout. Si el código C++ entraba en un bucle infinito o deadlock, el thread quedaba bloqueado indefinidamente, y aunque el event loop principal seguía vivo, la simulación se detenía y no se podía pausar ni detener limpiamente.

## Solución Implementada

### 1. Recarga Automática al Cambiar Grid Size

Se modificó `src/pipelines/handlers/inference_handlers.py` para detectar cambios en `grid_size`. Si se detecta un cambio y hay un experimento activo, se invoca automáticamente `handle_load_experiment` para recargar el motor con la nueva configuración.

```python
# src/pipelines/handlers/inference_handlers.py
if grid_size is not None and g_state.get('active_experiment'):
    logging.info(f"🔄 Recargando experimento... para aplicar nuevo grid size: {new_size}")
    # ...
    await handle_load_experiment(...)
```

### 2. Timeout en Simulation Loop

Se envolvió la llamada a `motor.evolve_internal_state` en `src/pipelines/core/simulation_loop.py` con `asyncio.wait_for` y un timeout de 5 segundos.

```python
# src/pipelines/core/simulation_loop.py
try:
    await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(None, motor.evolve_internal_state),
        timeout=5.0
    )
except asyncio.TimeoutError:
    logging.error("❌ Timeout crítico en motor.evolve_internal_state (5s)...")
    g_state['is_paused'] = True
    # ...
```

Esto asegura que si el motor nativo se bloquea, el backend recupera el control, pausa la simulación y notifica al usuario en lugar de quedarse congelado.

## Archivos Modificados

- `src/pipelines/handlers/inference_handlers.py`: Agregada lógica de recarga.
- `src/pipelines/core/simulation_loop.py`: Agregado timeout de seguridad.

## Verificación

- **Grid Size:** Al enviar un comando `set_inference_config` con `grid_size=32`, el sistema ahora recarga el modelo y el log debería mostrar `Grid escalado` (o la ausencia del mensaje si coincide con training size) y el motor inicializado con el nuevo tamaño.
- **Freeze:** Si el motor nativo se bloquea, después de 5 segundos la simulación se pausará automáticamente y aparecerá un mensaje de error en el frontend, permitiendo al usuario cambiar a motor Python o reiniciar.
