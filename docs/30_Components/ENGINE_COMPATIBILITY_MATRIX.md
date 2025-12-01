# Matriz de Compatibilidad de Motores

Este documento detalla las capacidades, soporte de física y características de visualización de los diferentes motores de simulación disponibles en Atheria.

## Resumen de Motores

| Motor | Tecnología | Rendimiento | Caso de Uso Principal |
|-------|------------|-------------|-----------------------|
| **Python (Standard)** | PyTorch (GPU/CPU) | Medio/Bajo | Prototipado, Física Lattice, Experimentos Flexibles |
| **Nativo (C++)** | C++ / OpenMP / TorchScript | Alto (CPU Bound) | Entrenamiento a gran escala, Inferencia rápida, Grids grandes |

## Soporte de Física

| Característica | Motor Python | Motor Nativo (C++) | Notas |
|----------------|--------------|--------------------|-------|
| **QCA Standard** | ✅ Completo | ✅ Completo | Misma lógica matemática (verificada). |
| **Lattice (SU3)** | ✅ Completo | ❌ No Soportado | Requiere motor Python para simulaciones de Gauge Theory. |
| **Harmonic (Ondas)** | ✅ Completo | ⚠️ Parcial | Python usa interferencia de ondas real. Nativo usa ruido determinista optimizado. |
| **Polar (Magnitud/Fase)** | ⚠️ En Desarrollo | ⚠️ Parcial | Soporte básico en ambos. |
| **Inyección de Física** | ✅ Completo | ✅ Completo | Soporta inyección de partículas y modificaciones de estado. |

## Soporte de Visualización

| Modo de Visualización | Motor Python | Motor Nativo (C++) | Notas |
|-----------------------|--------------|--------------------|-------|
| **Densidad** | ✅ Soportado | ✅ **Nativo (Rápido)** | Nativo calcula esto directamente en C++ sin overhead. |
| **Fase** | ✅ Soportado | ✅ **Nativo (Rápido)** | Nativo devuelve radianes o valores normalizados. |
| **Energía** | ✅ Soportado | ✅ **Nativo (Rápido)** | Cálculo de Hamiltoniano optimizado en C++. |
| **Flujo (Quiver)** | ✅ Soportado | ⚠️ **Fallback (Lento)** | Nativo usa fallback a Python (requiere conversión sparse->dense). |
| **Poincaré** | ✅ Soportado | ⚠️ **Fallback (Lento)** | Nativo usa fallback a Python. |
| **Phase Attractor** | ✅ Soportado | ⚠️ **Fallback (Lento)** | Nativo usa fallback a Python. |
| **Phase HSV** | ✅ Soportado | ⚠️ **Fallback (Lento)** | Nativo usa fallback a Python. |
| **Complex 3D** | ✅ Soportado | ⚠️ **Fallback (Lento)** | Nativo usa fallback a Python. |

> **Nota sobre Fallback:** El "Fallback" en el motor nativo implica transferir el estado denso de C++ a Python, lo cual puede ser costoso para grids grandes (>512x512). Se recomienda usar modos nativos para máximo rendimiento.

## Características Avanzadas

| Característica | Motor Python | Motor Nativo (C++) |
|----------------|--------------|--------------------|
| **ROI (Region of Interest)** | ❌ No necesario | ✅ Soportado | Crucial para visualizar detalles en grids gigantes sin procesar todo el estado. |
| **Lazy Conversion** | N/A | ✅ Soportado | Solo convierte sparse->dense cuando es estrictamente necesario. |
| **Checkpointing** | ✅ Soportado | ✅ Soportado | Compatible entre motores (formato .pt estándar). |
| **Entrenamiento** | ✅ Soportado | ✅ Soportado | El motor nativo es recomendado para entrenamientos largos. |

## Recomendaciones

1.  **Para Entrenamiento:** Usar **Motor Nativo**. Es mucho más rápido y eficiente en memoria.
2.  **Para Visualización Avanzada (Flow, Poincaré):** Usar **Motor Python** si el grid es pequeño (<256), o aceptar el rendimiento reducido del fallback en Nativo.
3.  **Para Física Lattice:** Usar **Motor Python** obligatoriamente.
4.  **Para Grids Gigantes (>1024):** Usar **Motor Nativo** con visualización de Densidad/Fase y ROI activado.
