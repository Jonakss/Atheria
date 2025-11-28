# Fix: Inference Persistence & Import Errors

**Fecha:** 2025-11-28
**Autor:** Antigravity (Assistant)
**Componentes:** Backend (Server Handlers, WebSocket Service), Frontend (WebSocketContext)

## Problema Detectado
El usuario reportó que la inferencia no persistía o no era recuperable tras una desconexión (ej. recargar la página).
Se identificaron dos causas principales:
1. **Pérdida de Estado de Experimento Activo:** El servidor no rastreaba qué experimento estaba cargado en `g_state['active_experiment']`. Al reconectar, el frontend recibía un estado inicial sin experimento activo, ocultando los controles y dando la apariencia de que la simulación se había detenido o perdido.
2. **Crash Silencioso por Imports Incorrectos:** La función `handle_load_experiment` en `server_handlers.py` contenía importaciones relativas incorrectas (`from . import config`) que causaban un `ImportError` silencioso (capturado por un bloque `try-except` genérico), impidiendo la carga correcta del modelo en algunos casos.

## Solución Implementada

### Backend
1. **Corrección de Imports:** Se corrigieron las importaciones relativas en `src/server/server_handlers.py` para apuntar correctamente a `src.config` y `src.model_loader`.
2. **Persistencia de `active_experiment`:**
   - Se actualiza `g_state['active_experiment']` al cargar un experimento exitosamente.
   - Se incluye `active_experiment` en el payload de `initial_state` enviado por `WebSocketService` a nuevas conexiones.

### Frontend
1. **Restauración de Estado:** En `WebSocketContext.tsx`, se actualizó la lógica para leer `active_experiment` del mensaje `initial_state` y restaurar el estado `activeExperiment` en el cliente. Esto asegura que la UI refleje correctamente que hay un experimento corriendo.

## Verificación
- Se creó un test de integración `tests/test_persistence.py` que verifica que `handle_load_experiment` actualiza correctamente `g_state['active_experiment']`.
- El test confirmó que las importaciones ahora funcionan correctamente y el estado se persiste.

## Impacto
- La simulación ahora es robusta a desconexiones y recargas de página.
- El usuario puede cerrar la pestaña y volver, y encontrará la simulación corriendo y controlable.
