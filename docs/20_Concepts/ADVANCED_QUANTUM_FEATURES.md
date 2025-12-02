# Advanced Quantum Features

Este documento detalla las características cuánticas avanzadas implementadas en Atheria, que transforman la simulación de un autómata celular pasivo a un sistema híbrido cuántico-clásico interactivo.

## 1. Quantum Tuner (Génesis Variacional)

El **Quantum Tuner** optimiza el estado inicial del universo para maximizar la complejidad emergente.

### Concepto
En lugar de iniciar con ruido aleatorio, utilizamos un **Circuito Cuántico Variacional (VQC)** en IonQ para generar el estado inicial. Los parámetros del circuito (ángulos de rotación) se ajustan mediante un algoritmo de optimización.

### Implementación
-   **Script**: `scripts/quantum_tuner.py`
-   **Optimizador**: SPSA (Simultaneous Perturbation Stochastic Approximation), ideal para funciones de costo ruidosas.
-   **Función Objetivo**: Maximizar $C = H \times S$
    -   $H$ (Entropía): Diversidad de estados.
    -   $S$ (Estabilidad): Inverso de la varianza de energía (persistencia).
-   **Salida**: `best_quantum_params.json` (ángulos óptimos).

---

## 2. Hybrid Compute (Colapso en Tiempo Real)

El **Hybrid Compute** integra el procesamiento cuántico *dentro* del bucle de simulación, no solo al inicio.

### Concepto
La simulación clásica "consulta" al procesador cuántico periódicamente. Una región del universo se codifica en qubits, evoluciona bajo una unitaria cuántica, y se "mide" (colapsa). El resultado de la medición actualiza el estado de la simulación.

### Implementación
-   **Módulo**: `src/physics/quantum_collapse.py` (`IonQCollapse`).
-   **Flujo**:
    1.  **Encoding**: Fase local $\rightarrow$ Rotación $R_y(\theta)$.
    2.  **Proceso**: Entrelazamiento (CNOTs) en IonQ.
    3.  **Decoding**: Medición $\rightarrow$ Magnitud del estado.
-   **Integración**: `CartesianEngine.evolve_hybrid_step` llama a `collapse()` cada N pasos.

---

## 3. Quantum Steering (Quantum Brush)

El **Quantum Steering** permite al usuario interactuar con la simulación inyectando patrones cuánticos específicos.

### Concepto
El usuario actúa como un "observador participativo", usando un "pincel cuántico" para pintar estados con propiedades físicas específicas (vorticidad, coherencia, entrelazamiento) en el grid.

### Implementación
-   **Módulo**: `src/physics/steering.py` (`QuantumSteering`).
-   **Patrones**:
    -   **Vortex**: Estado con fase rotacional (momento angular orbital).
    -   **Soliton**: Paquete de ondas localizado y coherente.
    -   **Entanglement**: Pares de Bell distribuidos.
-   **API**: `handle_interaction` en `server_handlers.py` procesa la acción `quantum_steer`.

---

## 4. Quantum Microscope (Deep Quantum Kernel)

El **Quantum Microscope** utiliza la ventaja cuántica para analizar estructuras ocultas en la simulación.

### Concepto
Mapea un parche del grid ($N \times N$) a un espacio de Hilbert de alta dimensión usando un **Deep Quantum Kernel**. Esto permite detectar correlaciones no locales (entrelazamiento, orden topológico) que son invisibles para kernels clásicos o redes neuronales estándar.

### Implementación
-   **Módulo**: `src/physics/quantum_kernel.py` (`QuantumMicroscope`).
-   **Circuito (Qiskit)**:
    1.  **Superposición**: Capa Hadamard.
    2.  **ZZ Feature Map**: Codifica la densidad del grid en correlaciones de fase ($R_{ZZ}$).
    3.  **RealAmplitudes (Ansatz)**: Capas de entrelazamiento fuerte y rotaciones.
    4.  **Medición**: Valores de expectación $\langle Z_i \rangle$.
-   **Métricas**:
    -   **Complejidad**: Varianza de las expectaciones (riqueza estructural).
    -   **Coherencia**: Pureza del estado.
    -   **Actividad**: Intensidad de las correlaciones.

---

## Arquitectura de Integración

```mermaid
graph TD
    User[Usuario / Frontend] -->|Steering (Brush)| API[Server API]
    User -->|Tuner (Script)| Tuner[Quantum Tuner]
    User -->|Vision (Microscope)| API

    subgraph Atheria Engine
        Sim[Cartesian Engine] <-->|Hybrid Step| Collapse[IonQCollapse]
        Sim <-->|Injection| Steering[Quantum Steering]
        Sim -->|Analysis| Microscope[Quantum Microscope]
    end

    Tuner -->|Params| Sim

    Collapse <-->|API| IonQ[IonQ Backend / Simulator]
    Steering <-->|API| IonQ
    Tuner <-->|API| IonQ
    Microscope <-->|API| IonQ
```
