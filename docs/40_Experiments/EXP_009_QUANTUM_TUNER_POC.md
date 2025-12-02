# EXP-009: Prueba de Concepto - Sintonizador de Hiperparámetros Cuántico (Quantum Tuner)

**Estado:** Diseño / PoC
**Hardware:** IBM Quantum (via Qiskit Runtime)

## Objetivo
Implementar un script que utilice un procesador cuántico para optimizar los hiperparámetros de Aetheria (`GAMMA_DECAY`, `LR_RATE`), demostrando la ventaja de los algoritmos variacionales para la exploración de espacios complejos.

## Arquitectura del Experimento

### 1. Circuito Variacional (Ansatz)
Un circuito cuántico simple actúa como el "mapa" de los parámetros.
* **Qubits:** 2
* **Parámetros:** $\theta_0$ (controla `GAMMA`), $\theta_1$ (controla `LR`).
* **Puertas:** Rotaciones $R_y(\theta)$ y entrelazamiento $CZ$.

### 2. Función de Costo Híbrida
```python
def evaluate_params(theta):
    # 1. Decodificar parámetros
    gamma = map_range(theta[0], 0, 1)
    lr = map_range(theta[1], 1e-5, 1e-3)

    # 2. Correr simulación clásica rápida (GPU)
    sim_result = run_aetheria_snippet(gamma, lr, steps=50)

    # 3. Calcular "Interesancia" (Entropía)
    score = calculate_entropy(sim_result)

    return -score # Minimizar el negativo
```

### 3. Ejecución en Qiskit Runtime
Uso de `EstimatorV2` y el optimizador `SPSA`.
```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Session
from qiskit_algorithms.optimizers import SPSA

# ... configuración ...
with Session(service=service, backend=backend) as session:
    estimator = EstimatorV2(session=session)
    # Bucle de optimización
    result = spsa.minimize(evaluate_params, x0=[0.5, 0.5])
```

## Resultados Esperados
* Encontrar una configuración de parámetros que genere patrones estables (vida) en menos iteraciones que una búsqueda aleatoria o grid search.
* Validar el flujo de trabajo híbrido QPU-GPU.
