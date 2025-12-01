# 2025-12-01 - Fix: Native Engine JIT Export Freeze

## Problema
El usuario reportó que la simulación se congelaba y desconectaba al intentar cargar un experimento con el motor nativo.
El análisis de logs mostró:
1.  `export_model_to_jit` tomaba ~45 segundos, bloqueando el hilo principal (asyncio loop).
2.  Esto causaba que el servidor dejara de responder a heartbeats, provocando `ClientConnectionResetError`.
3.  Además, la inicialización del estado denso (`_initialize_native_state_from_dense`) realizaba 65k iteraciones en Python sin liberar el GIL explícitamente, lo que podría causar starvation del hilo principal incluso corriendo en un thread separado.

## Solución
1.  **Offloading de JIT Export:** Se movió la lógica de exportación JIT en `inference_handlers.py` a una función interna ejecutada con `loop.run_in_executor`. Esto libera el event loop para mantener la conexión WebSocket viva.
2.  **Optimización de Inicialización:** Se añadieron puntos de liberación de GIL (`time.sleep(0)`) cada 10 filas en el bucle de inicialización de partículas en `native_engine_wrapper.py`. También se añadieron logs de timing para monitorear el rendimiento de esta fase crítica.

## Archivos Modificados
- `src/pipelines/handlers/inference_handlers.py`: Offloading de `export_model_to_jit`.
- `src/engines/native_engine_wrapper.py`: `time.sleep(0)` y logs en `add_particles_with_threshold`.

## Verificación
- El servidor ahora debería responder a pings/heartbeats mientras exporta el modelo, evitando la desconexión.
- La inicialización del motor nativo debería ser más "amigable" con el event loop.
