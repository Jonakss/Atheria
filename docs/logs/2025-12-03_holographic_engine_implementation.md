---
type: log
date: 2025-12-03
tags: [feature, engine, holographic, ads-cft]
related: [[HOLOGRAPHIC_PRINCIPLE], [LatticeEngine], [CartesianEngine]]
---

# 2025-12-03 - Feature: Holographic Engine (AdS/CFT Projection)

## 🎯 Objetivo
Implementar un nuevo motor de física (`HolographicEngine`) que materialice el **Principio Holográfico** en Atheria. El objetivo es permitir que un estado 2D (Frontera) genere un volumen 3D (Bulk) emergente, proporcionando una base para futuras visualizaciones volumétricas y experimentos de gravedad emergente.

## 🛠️ Implementación Técnica

### 1. `HolographicEngine` (`src/engines/holographic_engine.py`)
Se creó una nueva clase que hereda de `CartesianEngine`. Esto significa que la dinámica fundamental sigue siendo la de un QCA 2D (compatible con todos los modelos y herramientas existentes), pero añade capacidades de proyección.

**Método Clave: `get_bulk_state()`**
La proyección del Boundary (2D) al Bulk (3D) se implementó utilizando una técnica de **Scale-Space (Espacio de Escala)**.
*   **Teoría:** En la correspondencia AdS/CFT, la dimensión radial extra ($Z$) está relacionada con la escala de energía o renormalización. Los objetos profundos en el bulk corresponden a excitaciones de baja frecuencia (gran escala) en la frontera.
*   **Algoritmo:**
    *   La capa $Z=0$ es el estado original (magnitud/energía).
    *   Para capas $Z > 0$, aplicamos un **Gaussian Blur** progresivo.
    *   $\sigma = 0.5 \cdot Z + 0.5$
    *   Esto filtra las altas frecuencias, dejando solo las estructuras grandes en las capas profundas.

### 2. Integración en `MotorFactory`
Se actualizó `src/motor_factory.py` para reconocer el tipo de motor `HOLOGRAPHIC`.
```python
elif engine_type == 'HOLOGRAPHIC':
    logging.info("🔮 Initializing Holographic Engine (AdS/CFT Projection)")
    return HolographicEngine(model, grid_size, d_state, backend.get_device(), cfg=config)
```

## 🧪 Verificación

Se creó el test `tests/test_holographic_engine.py` para verificar:
1.  **Inicialización:** Correcta herencia y configuración.
2.  **Proyección:** Generación de un tensor volumétrico `[1, D, H, W]`.
3.  **Propiedad Holográfica:** Se verificó que la varianza de la señal disminuye con la profundidad ($Z$), confirmando que la información se "suaviza" o "renormaliza" hacia el interior del bulk.

**Resultados del Test:**
```
Variance Layer 0: 0.2376
Variance Layer 7: 0.0016
All tests passed!
```

## 📝 Siguientes Pasos
1.  **Visualización Frontend:** Crear un componente `HolographicVolumeViewer` en React/Three.js que pueda renderizar este volumen (Texture3D o Raymarching).
2.  **Entropía:** Implementar cálculo de entropía de entrelazamiento (Ryu-Takayanagi) usando las superficies mínimas en este bulk generado.
