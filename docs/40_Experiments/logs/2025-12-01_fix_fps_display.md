# 2025-12-01 - Fix: FPS Display in SimulationService

## Contexto
El contador de FPS en la UI mostraba constantemente "N/A" o "0.0". Esto se debía a una regresión introducida al migrar del bucle de simulación legacy (`simulation_loop.py`) a la nueva arquitectura de servicios (`SimulationService`).

## Problema
`SimulationService` es responsable de ejecutar los pasos de física, pero no estaba calculando ni actualizando la métrica `current_fps` en el estado global (`g_state`). Aunque `DataProcessingService` leía esta métrica para enviarla al frontend, el valor nunca se actualizaba desde el origen.

## Solución
Se implementó la lógica de cálculo de FPS en `src/services/simulation_service.py`:
1.  Se añadieron contadores (`fps_counter`) y temporizadores (`fps_start_time`) en la inicialización del servicio.
2.  En el bucle `_simulation_loop`, se incrementa el contador en cada paso.
3.  Cada 0.5 segundos, se calcula el FPS real (`pasos / tiempo`) y se actualiza `g_state['current_fps']`.

## Archivos Modificados
- `src/services/simulation_service.py`: Implementación del cálculo de FPS.

## Verificación
- Se verificó manualmente en la UI que el contador de FPS ahora muestra valores numéricos fluctuantes (ej: 60.0) en lugar de "N/A".
