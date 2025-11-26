### 2025-11-26: Mejora en la Visualización de Pasos de Simulación

**Agente:** Gemini-CLI

**Descripción de Cambios:**
Se ha mejorado la interfaz de usuario del frontend para proporcionar información más detallada sobre el progreso de la simulación. La barra de herramientas ahora muestra el paso inicial, el paso actual y el total de pasos configurados, siguiendo el formato `inicio / actual (total)`.

1.  **Backend (`simulation_loop.py`, `inference_handlers.py`):**
    -   Se modificaron los payloads de WebSocket (`simulation_frame` y `simulation_state_update`) para incluir `start_step` y `total_steps`.
    -   Se aseguró que `start_step` se reinicie a `0` cada vez que se carga un nuevo experimento para mantener la consistencia.

2.  **Frontend (`Toolbar.tsx`):**
    -   Se actualizó el componente de la barra de herramientas para consumir y mostrar los nuevos campos del estado de la simulación.
    -   El tooltip sobre el contador de pasos ahora muestra el progreso como un porcentaje para una comprensión más rápida del estado de la simulación.

**Impacto:**
Esta mejora proporciona al usuario una visión mucho más clara y contextual del avance de la simulación, especialmente útil en ejecuciones largas o cuando se retoma desde un checkpoint.

**Próximos Pasos:**
-   Permitir al usuario configurar `total_steps` desde la interfaz de usuario.