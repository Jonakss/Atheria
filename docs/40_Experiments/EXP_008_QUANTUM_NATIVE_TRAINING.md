# EXP-008: Quantum-Native Training for Massive Fast Forward

**Fecha:** 2025-12-04
**Autores:** Aetheria AI Team
**Estado:** ✅ Completado (Proof of Concept)

## Abstract
Este experimento aborda el problema de escalabilidad encontrado en `EXP-007`, donde la implementación de operadores de evolución temporal ($U^N$) mediante compuertas diagonales arbitrarias resultó en un costo exponencial de compuertas ($O(2^N)$), inviable para hardware cuántico ruidoso (NISQ). Proponemos y evaluamos un enfoque de **Entrenamiento Cuántico Nativo**, donde una red neuronal parametrizada (PQC) basada en compuertas eficientes ($R_z$, $R_{zz}$) es entrenada para aproximar el operador objetivo. Los resultados demuestran una reducción drástica en la profundidad del circuito ($O(N \cdot L)$), aunque revelan un compromiso significativo en la expresividad (Fidelidad $\approx 1\%$) para operadores altamente complejos con topologías de entrelazamiento simples.

## 1. Introducción
La simulación de sistemas complejos mediante "Massive Fast Forward" requiere elevar un operador de evolución $U$ a una gran potencia $N$. En el dominio de la frecuencia (Holografía), esto equivale a una matriz diagonal de fases. Sin embargo, en computación cuántica, una diagonal arbitraria no es una operación nativa. Su descomposición en compuertas estándar (CNOTs + Rotaciones) escala exponencialmente con el número de qubits, limitando severamente la aplicabilidad en dispositivos como IonQ Aria/Harmony.

El objetivo de este trabajo es validar si un Circuito Cuántico Parametrizado (PQC) diseñado con compuertas nativas de hardware puede aprender a aproximar este operador complejo, manteniendo una profundidad de circuito constante y manejable.

## 2. Metodología

### 2.1. Operador Objetivo ($W_{target}$)
Se extrajo el operador de evolución efectivo de una UNet unitaria pre-entrenada (`EXP-007`), correspondiente a $10^6$ pasos de tiempo.
- **Grid:** $32 \times 32$ (10 Qubits).
- **Naturaleza:** Matriz diagonal de fases complejas en el dominio de Fourier.

### 2.2. Arquitectura Cuántica Nativa (Ansatz)
Se diseñó un PQC `QuantumNativeConv2d` optimizado para trampas de iones:
- **Capas ($L$):** 10.
- **Compuertas Locales:** $R_z(\theta)$ en cada qubit (Costo 0 CNOTs).
- **Entrelazamiento:** $R_{zz}(\phi)$ entre vecinos lineales (Costo 1 CNOT efectiva en Mølmer-Sørensen).
- **Profundidad Total:** $O(L)$ independiente del tamaño del grid.

### 2.3. Protocolo de Entrenamiento
- **Optimizador:** Adam (LR=0.01).
- **Función de Pérdida:** MSE en el plano complejo entre las fases generadas por el PQC y el Target.
- **Simulación:** PyTorch simulando la mecánica cuántica (diferenciable).

## 3. Resultados

### 3.1. Eficiencia de Hardware (Gate Count)
| Métrica | Diagonal Arbitraria (EXP-007) | Quantum Native (EXP-008) | Mejora |
| :--- | :--- | :--- | :--- |
| **CNOTs (10 Qubits)** | ~1,024 | ~90 (10 capas) | **11x** |
| **CNOTs (16 Qubits)** | ~65,536 (Fallo) | ~150 (10 capas) | **436x** |
| **Escalabilidad** | Exponencial $O(2^N)$ | Lineal $O(N \cdot L)$ | **Exponencial** |

### 3.2. Fidelidad y Expresividad
A pesar de la ventaja en eficiencia, el modelo alcanzó una fidelidad baja:
- **Fidelidad Final:** $\approx 0.01$ (1%)
- **Interpretación:** El ansatz lineal de vecinos más cercanos ($R_{zz}$ lineal) no tiene suficiente "conectividad" o expresividad para capturar la estructura de fase altamente no-local y "aleatoria" que generó la UNet clásica. El operador objetivo es demasiado complejo (alta entropía de fase) para ser comprimido en un circuito tan simple.

## 4. Discusión
El experimento confirma que el enfoque "Nativo" resuelve el problema de ingeniería (profundidad del circuito), permitiendo correr modelos en grids grandes (20+ qubits) en hardware actual. Sin embargo, traslada el problema a la **arquitectura del modelo**:
- Un ansatz simple no basta para imitar una red neuronal clásica profunda.
- **Solución Futura:** En lugar de intentar que el PQC imite a la UNet clásica (Distillation), deberíamos **entrenar el PQC desde cero** en la tarea física original (End-to-End). De esta forma, la IA aprenderá una solución que *sí* quepa en el ansatz disponible, en lugar de forzar una solución clásica compleja en un molde cuántico simple.

## 5. Conclusión
El entrenamiento cuántico nativo es viable y necesario para escalar. Hemos generado circuitos QASM válidos y eficientes. El siguiente paso lógico no es aumentar capas (lo que reintroduce ruido), sino cambiar la estrategia de entrenamiento a "Quantum-First", donde el hardware define las restricciones desde el inicio del aprendizaje.

## 6. Artefactos
- **Modelo:** `checkpoints/quantum_native_model.pt`
- **Script:** `scripts/experiment_quantum_native_training.py`
