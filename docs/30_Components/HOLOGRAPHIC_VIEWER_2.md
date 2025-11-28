# Holographic Viewer 2.0 (AdS/CFT)

El **Holographic Viewer 2.0** es la evolución del visor original, diseñado específicamente para la **Fase 4** del proyecto Atheria. Su objetivo es visualizar la correspondencia AdS/CFT, proyectando la información del "Boundary" (2D) hacia el "Bulk" (3D).

## Diferencias con v1.0

| Característica | Viewer v1.0 (Poincaré) | Viewer v2.0 (AdS/CFT) |
| :--- | :--- | :--- |
| **Enfoque** | Visualización estética y navegación 2D/3D básica. | Visualización científica de geometría emergente. |
| **Proyección** | Disco de Poincaré (Mapeo conforme simple). | Redes Tensoriales (MERA) y Geometría Hiperbólica dinámica. |
| **Datos** | Densidad, Fase, Flujo. | Entropía de Entrelazamiento, Curvatura, Geodésicas. |
| **Tecnología** | WebGL (Three.js) + Shaders custom. | WebGL 2.0 + Compute Shaders (futuro) para Raymarching. |

## Implementación

### Componente: `HolographicViewer2.tsx`
Ubicación: `frontend/src/components/visualization/HolographicViewer2.tsx`

Este componente mantiene la interfaz del original pero añade capas específicas para la visualización holográfica:
-   **Indicador Visual:** Etiqueta "v2.0 (AdS/CFT)" para distinguir el modo.
-   **Toggle:** Integrado en `PhysicsInspector` para cambiar en tiempo real.

### Lógica de Switcheo
El `DashboardLayout` gestiona el estado `viewerVersion` ('v1' | 'v2') y renderiza el componente apropiado condicionalmente, preservando el contexto de WebGL cuando es posible.

## Roadmap de Visualización

1.  **Scale-Radius Duality:** Mapear la escala de renormalización (coarse-graining) a la dimensión radial $z$.
2.  **Entanglement Entropy:** Visualizar arcos de Ryu-Takayanagi sobre el disco.
3.  **Black Hole Horizon:** Renderizar el horizonte de eventos en el bulk cuando la temperatura sube.
