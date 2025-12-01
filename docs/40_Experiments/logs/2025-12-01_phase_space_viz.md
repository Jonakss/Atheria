# 2025-12-01 - Feature: Phase Space Visualization (PCA + Clustering)

## Contexto
Para entender mejor la topología del espacio de estados de Aetheria, necesitamos visualizar cómo se agrupan los estados de las células en un espacio de características reducido. Esto nos permite identificar "tipos de materia" emergentes (vacío, paredes, partículas) de forma automática.

## Cambios Implementados

### Backend
- **Nuevo Módulo:** `src/pipelines/viz/phase_space.py`
    - Implementa `get_phase_space_data(psi, n_clusters=3)`.
    - **PCA:** Reduce la dimensionalidad de `d_state` (complejo -> 2*real) a 3 componentes principales.
    - **K-Means:** Agrupa los puntos en el espacio reducido para identificar clusters.
    - **Optimizaciones:**
        - **Subsampling:** Usa un stride dinámico para limitar el análisis a ~10k puntos, manteniendo el rendimiento en tiempo real.
        - **Caché:** Evita recalcular PCA/K-Means si el estado no ha cambiado significativamente (hash check).
- **Integración:** Actualizado `src/pipelines/viz/core.py` para incluir el nuevo tipo de visualización `phase_space`.

### Frontend
- **Nueva Opción:** Añadido "Espacio de Fases (PCA)" a `vizOptions.ts`.

## Por qué
- **PCA:** Permite visualizar correlaciones entre canales que no son evidentes en la vista espacial directa.
- **K-Means:** Automatiza la clasificación de células, permitiendo al sistema "descubrir" tipos de estructuras sin reglas predefinidas.

## Verificación
- Tests unitarios en `tests/test_viz_phase_space.py` verifican la estructura de datos, la reducción de dimensionalidad y el caching.
