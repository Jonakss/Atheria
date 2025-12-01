# Visualización del Espacio de Fases (PCA + Clustering)

## 🌌 Concepto

La **Visualización del Espacio de Fases** es una técnica analítica avanzada en Atheria que permite observar la topología del estado cuántico del universo más allá de su representación espacial directa.

Mientras que la vista normal nos muestra la distribución espacial de la energía o la fase en el grid 2D, el espacio de fases nos revela cómo se relacionan los diferentes canales del estado cuántico (`d_state`) entre sí, independientemente de su posición en el espacio.

## 🧠 Metodología

El proceso transforma el tensor de estado $\psi$ de dimensiones $[H, W, d_{state}]$ en una nube de puntos 3D mediante los siguientes pasos:

1.  **Aplanado (Flattening):** Cada celda del grid se trata como una muestra individual en un espacio de $d_{state}$ dimensiones.
2.  **Manejo de Complejos:** Dado que el estado es complejo, se concatenan la parte real e imaginaria, resultando en un espacio de características de $2 \times d_{state}$ dimensiones.
3.  **Reducción de Dimensionalidad (PCA):** Se aplica **Análisis de Componentes Principales** para proyectar este espacio de alta dimensión en un espacio 3D (las 3 componentes con mayor varianza).
4.  **Clustering (K-Means):** Se agrupan los puntos en el espacio reducido para identificar estructuras emergentes automáticamente.

## 🔍 Interpretación Física

Esta visualización nos permite identificar "tipos de materia" emergentes:

*   **Cluster 0 (Vacío):** Puntos cercanos al origen (0,0,0). Representan el vacío cuántico o estado base.
*   **Cluster 1 (Paredes/Estructuras):** Puntos que forman estructuras estables o topológicas.
*   **Cluster 2 (Excitaciones/Partículas):** Puntos alejados del origen o con características espectrales únicas, representando excitaciones energéticas.

## ⚙️ Implementación Técnica

La implementación se encuentra en `src/pipelines/viz/phase_space.py` y utiliza:
*   `scikit-learn` para PCA y K-Means.
*   **Subsampling Inteligente:** Analiza solo una fracción de los puntos (stride dinámico) para mantener el rendimiento en tiempo real (~60 FPS).
*   **Caché:** Evita recalcular la proyección si el estado cuántico no ha cambiado significativamente.

---

## Enlaces Relacionados

- [[QUALITY_DIVERSITY_MAP_ELITES]] - Algoritmo MAP-Elites que explora el espacio de fases
- [[NEURAL_CELLULAR_AUTOMATA_THEORY]] - Teoría NCA y análisis de estado
- [[3D_STATE_SPACE_CONCEPT]] - Conceptualización del espacio de estados

**Tags:** #concept #visualization #physics #pca #clustering #phase-space
