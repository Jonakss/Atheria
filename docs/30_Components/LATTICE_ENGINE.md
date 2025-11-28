# Lattice Engine (Phase 4)

El **Lattice Engine** es el componente central de la Fase 4 de Atheria, diseñado para simular **Lattice Gauge Theory** (Teoría Cuántica de Campos en Retículo).

## Propósito
A diferencia de los motores anteriores (Sparse, Native) que se enfocan en la evolución de funciones de onda (Schrödinger/Lindblad), el Lattice Engine simula la dinámica de **campos de gauge** (como gluones en QCD) que viven en los enlaces (links) de un retículo espacio-temporal.

## Arquitectura

### Estructura del Retículo
-   **Nodos (Sitios):** Representan puntos en el espacio-tiempo. Pueden contener campos de materia (fermiones).
-   **Enlaces (Links):** Conectan nodos adyacentes. Representan el campo de gauge $U_\mu(x)$.
    -   Son elementos de un grupo de Lie, típicamente $SU(N)$.
    -   Para $SU(3)$, son matrices complejas unitarias $3 \times 3$.

### Acción de Wilson
La dinámica se rige por la **Acción de Wilson**, que se calcula sumando la "plaqueta" (el producto de links alrededor de un cuadrado elemental):

$$ S = -\beta \sum_{p} \text{Re}(\text{Tr}(U_p)) $$

Donde $U_p$ es la plaqueta.

### Algoritmo de Evolución
El motor utiliza algoritmos de Monte Carlo (como **Metropolis-Hastings** o **Heat Bath**) para generar configuraciones de campo que sigan la distribución de probabilidad cuántica $P(U) \propto e^{-S(U)}$.

## Implementación (`src/engines/lattice_engine.py`)

```python
class LatticeEngine:
    def __init__(self, grid_size, d_state, group='SU3', beta=6.0):
        # ...
```

## Integración
El Lattice Engine se integra en el pipeline de inferencia de Atheria como una opción de motor seleccionable (`force_engine="lattice"`).
