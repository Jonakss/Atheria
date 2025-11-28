# 2025-11-28 - Poincaré Projection & Quadtree Optimization

## Resumen
Se implementó la visualización del **Disco de Poincaré** en el `HolographicViewer` y se optimizó el renderizado de **Quadtrees** en el frontend para evitar bloqueos del navegador con grids grandes.

## Cambios Realizados

### Frontend
- **HolographicViewer.tsx**:
    - Se agregó la prop `vizType` para controlar el tipo de visualización.
    - Se implementó la transformación matemática de coordenadas cuadradas a disco hiperbólico cuando `vizType === 'poincare'`.
    - Se mapea la magnitud a $Z$ (altura) y el tamaño de partícula.
- **CanvasOverlays.tsx**:
    - Se implementó **Viewport Culling** (Recorte de Vista) para el renderizado del Quadtree.
    - Ahora solo se procesan y dibujan los nodos del quadtree que son visibles en la pantalla, basándose en el zoom y pan actual.
    - Se eliminó el límite duro que deshabilitaba el quadtree para grids grandes (>256x256), ya que la optimización permite manejar grids mucho mayores siempre que el usuario haga zoom.
- **DashboardLayout.tsx**:
    - Se actualizó para pasar la prop `vizType` (basada en `selectedViz`) al componente `HolographicViewer`.

### Documentación
- **ROADMAP_PHASE_4.md**:
    - Se marcó la tarea "Disco de Poincaré" como completada.

## Impacto
- **Visualización:** Ahora es posible visualizar la simulación en una geometría hiperbólica, un paso clave hacia la correspondencia AdS/CFT.
- **Rendimiento:** La navegación en grids grandes con el overlay de Quadtree activado debería ser mucho más fluida y no causar bloqueos del navegador.
