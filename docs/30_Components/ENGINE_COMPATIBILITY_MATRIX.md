# Matriz de Compatibilidad de Motores

Este documento detalla las capacidades, soporte de física y características de visualización de los diferentes motores de simulación disponibles en Atheria.

## Resumen de Motores

| Motor | Tecnología | Rendimiento | Caso de Uso Principal |
|-------|------------|-------------|-----------------------|
| **Python (Standard)** | PyTorch (GPU/CPU) | Medio/Bajo | Prototipado, Física Lattice, Experimentos Flexibles |
| **Nativo (C++)** | C++ / OpenMP / TorchScript | Alto (CPU Bound) | Entrenamiento a gran escala, Inferencia rápida, Grids grandes |

## Engine Implementations Status

### Native Engine (C++)
- **Status**: Production-ready
- **Physics**: Standard QCA (Quantum Cellular Automata)
- **Performance**: 10-100x faster than Python
- **Limitations**: 
  - Advanced visualizations fallback to Python (Flow, Poincaré, Phase HSV)
  - Only supports standard Cartesian QCA physics

### Python Engines

#### Standard (CARTESIAN)
- **Status**: Production-ready
- **Physics**: Standard QCA
- **Backend**: Both Python and C++ (via Native Engine)
- **Use case**: Default for training and inference

#### Harmonic (HARMONIC)
- **Status**: Experimental (Python-only)
- **Physics**: Spectral methods, wave interference, FFT
- **Backend**: Python only (C++ implementation planned)
- **Use case**: Wave-based simulations, spectral analysis

#### Lattice (LATTICE)
- **Status**: Experimental (Python-only)
- **Physics**: SU(3) gauge theory, Wilson Action, AdS/CFT
- **Backend**: Python only (C++ implementation planned)
- **Use case**: Gauge theory simulations, holographic duality

#### Polar (POLAR)
- **Status**: Experimental (Python-only)
- **Physics**: Rotational coordinates, angular momentum
- **Backend**: Python only (C++ implementation planned)
- **Use case**: Rotational dynamics, vortex simulations

#### Quantum (QUANTUM)
- **Status**: Prototype (Python-only)
- **Physics**: Hybrid quantum-classical
- **Backend**: Python only
- **Use case**: Quantum algorithm research

### Roadmap: C++ Implementations

The following engines are planned for C++ implementation to achieve Native Engine performance:

1. **Harmonic Engine (C++)** - Priority: Medium
   - Port FFT and spectral methods to C++
   - Est. performance improvement: 50-100x

2. **Lattice Engine (C++)** - Priority: Low
   - Port SU(3) gauge field logic to C++
   - Est. performance improvement: 30-50x

3. **Polar Engine (C++)** - Priority: Medium
   - Port polar coordinate transform to C++
   - Est. performance improvement: 40-80x

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
