# API Commands Reference

This document is auto-generated. It lists all available WebSocket commands grouped by scope.

## Scope: `experiment`

| Command | Description |
|---------|-------------|
| `create` | Crea un nuevo experimento de entrenamiento. |
| `continue` | Continúa el entrenamiento de un experimento existente. |
| `stop` | Detiene el entrenamiento en curso. |
| `delete` | Elimina un experimento completo. |
| `list_checkpoints` | Lista todos los checkpoints de un experimento. |
| `delete_checkpoint` | Elimina un checkpoint específico. |
| `cleanup_checkpoints` | Limpia checkpoints antiguos de un experimento. |
| `refresh_experiments` | Refresca la lista de experimentos. |

## Scope: `simulation`

| Command | Description |
|---------|-------------|
| `set_viz` | Cambia el tipo de visualización. |
| `update_visualization` | Actualiza manualmente la visualización (útil para modo no-live). |
| `set_speed` | Controla la velocidad de la simulación (multiplicador). |
| `set_fps` | Controla los FPS objetivo de la simulación. |
| `set_frame_skip` | Controla cuántos frames saltar para acelerar (0 = todos, 1 = cada otro, etc.). |
| `set_live_feed` | Activa o desactiva el live feed (envío automático de frames). |
| `set_steps_interval` | Configura el intervalo de pasos para el envío de frames cuando live_feed está DESACTIVADO. |
| `set_compression` | Habilita o deshabilita la compresión de datos (gzip/zlib). |
| `set_downsample` | Configura el factor de downsampling para la visualización. |
| `set_roi` | Configura la región de interés (ROI) para visualización. |
| `set_snapshot_interval` | Configura el intervalo de captura de snapshots. |
| `enable_snapshots` | Habilita o deshabilita la captura automática de snapshots. |
| `capture_snapshot` | Captura un snapshot manual del estado actual de la simulación. |

## Scope: `inference`

| Command | Description |
|---------|-------------|
| `play` | Inicia la simulación. |
| `pause` | Pausa la simulación. |
| `load_experiment` | Carga un experimento. |
| `unload_model` | Descarga el modelo cargado y limpia el estado. |
| `switch_engine` | Cambia entre motor nativo (C++) y motor Python. |
| `reset` | Reinicia el estado de la simulación al estado inicial. |
| `inject_energy` | Inyecta energía en el estado cuántico actual. |
| `set_inference_config` | Configura parámetros de inferencia. |
| `set_config` | Configura parámetros de inferencia. |
| `set_viz` | Cambia el tipo de visualización. |

## Scope: `system`

| Command | Description |
|---------|-------------|
| `shutdown` | Apaga el servidor desde la UI. |
| `refresh_experiments` | Actualiza la lista de experimentos disponibles. |
| `toggle_logs` | Activa o desactiva el streaming de logs. |

## Scope: `analysis`

| Command | Description |
|---------|-------------|
| `universe_atlas` | Crea un "Atlas del Universo" analizando la evolución temporal usando t-SNE. |
| `cell_chemistry` | Crea un "Mapa Químico" analizando los tipos de células en el estado actual usando t-SNE. |
| `cancel` | Cancela cualquier análisis en curso. |
| `clear_snapshots` | Limpia todos los snapshots almacenados. |

## Scope: `history`

| Command | Description |
|---------|-------------|
| `enable_history` | Habilita o deshabilita el guardado de historia de simulación. |
| `save_history` | Guarda el historial de simulación a un archivo. |
| `clear_history` | Limpia el historial de simulación. |
| `list_history_files` | Lista los archivos de historia guardados. |
| `load_history_file` | Carga un archivo de historia. |
| `get_history_range` | Obtiene el rango de steps disponibles en el buffer de historia para navegación en vivo. |
| `restore_history_step` | Restaura el estado de la simulación a un step específico del historial. |

## Scope: `snapshot`

| Command | Description |
|---------|-------------|
| `save_snapshot` | Guarda el estado actual de la simulación. |
| `list_snapshots` | Lista los snapshots disponibles para un experimento. |
| `load_snapshot` | Carga un snapshot y restaura el estado de la simulación. |

