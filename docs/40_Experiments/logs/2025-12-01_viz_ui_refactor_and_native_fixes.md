# 2025-12-01 - Refactor: Visualization UI & Native Engine Fixes

## Resumen

Se ha completado una refactorización mayor de la UI de visualización y correcciones críticas para el soporte del Motor Nativo (C++).

## Cambios Principales

### 1. Refactorización de UI de Visualización

- **Nuevo Panel:** Se creó `VisualizationPanel.tsx` en el sidebar derecho.
- **Limpieza:** Se movieron los controles de visualización desde `PhysicsInspector.tsx` para evitar duplicidad y mejorar la organización.
- **Integración:** `DashboardLayout.tsx` actualizado para incluir el nuevo panel.

### 2. Soporte de Visualización para Motor Nativo

- **Backend:** Actualizado `src/pipelines/viz/core.py` para generar datos avanzados (`flow`, `poincare`, `phase_hsv`) incluso cuando se usa el motor nativo (o usar fallback a Python).
- **Frontend:** Implementados fallbacks robustos en `PanZoomCanvas.tsx`.
  - Si faltan datos (ej. limitación del motor nativo), se muestra un mensaje claro ("Flow data unavailable") en lugar de una pantalla en blanco.

### 3. Selección de Motor y Emergencia de Partículas

- **Inferencia:** Refactorizado `handle_load_experiment` para respetar estrictamente `ENGINE_TYPE` de la configuración del experimento.
- **Documentación:** Creado `docs/30_Components/PARTICLE_EMERGENCE.md` explicando la lógica de inicialización de partículas desde estados densos.
- **Matriz de Compatibilidad:** Creado `docs/30_Components/ENGINE_COMPATIBILITY_MATRIX.md` detallando soporte de features por motor.

## Archivos Afectados

- `frontend/src/modules/Dashboard/components/VisualizationPanel.tsx` (Nuevo)
- `frontend/src/components/ui/PanZoomCanvas.tsx` (Modificado)
- `src/pipelines/viz/core.py` (Modificado)
- `src/pipelines/handlers/inference_handlers.py` (Modificado)
- `docs/30_Components/ENGINE_COMPATIBILITY_MATRIX.md` (Nuevo)
- `docs/30_Components/PARTICLE_EMERGENCE.md` (Nuevo)
