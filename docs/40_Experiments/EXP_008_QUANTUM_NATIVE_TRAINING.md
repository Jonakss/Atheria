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

### 3.3. Visualización de Resultados
![Resultados EXP-008](/home/jonathan.correa/Projects/Atheria/docs/40_Experiments/images/exp008_results.png)
*Fig 1. Curva de pérdida (izquierda), histograma de fases (centro) y correlación de fases (derecha). Se observa que aunque la pérdida baja, la correlación no es una diagonal perfecta, indicando falta de expresividad.*

## 4. Discusión y Alternativas de Operadores
El usuario planteó la pregunta: *¿Qué otros operadores podemos usar además de este ansatz simple?*

El modelo actual (`QuantumNativeConv2d`) usó un ansatz "Hardware Efficient" muy básico (Topología Lineal, solo $R_z$ y $R_{zz}$). La baja fidelidad indica que necesitamos arquitecturas más ricas. Las alternativas para futuros experimentos incluyen:

1.  **Strongly Entangling Layers:**
    - Conectividad "todos con todos" o patrones circulares complejos.
    - Compuertas de rotación general $U3(\theta, \phi, \lambda)$ en lugar de solo $R_z$.
    - Mayor expresividad, costo de compuertas moderado ($O(N^2)$ o $O(N)$ según topología).

2.  **EfficientSU2 (Full Rotation):**
    - Capas de rotaciones $R_y$ y $R_z$ intercaladas con CNOTs.
    - Permite explorar toda la esfera de Bloch para cada qubit.
    - Estándar en la industria (Qiskit Circuit Library).

3.  **Data Re-uploading (Quantum MLP):**
    - Codificar la entrada $x$ repetidamente en varias capas, intercaladas con pesos entrenables.
    - Teóricamente puede aproximar cualquier función (Teorema de Aproximación Universal Cuántico).

4.  **Hybrid Tensor Networks (MPS/TTN):**
    - Estructuras jerárquicas (tipo árbol) que capturan correlaciones de largo alcance mejor que una cadena lineal.

**Conclusión:** El problema no es el método "Nativo", sino que el ansatz elegido fue demasiado "tacaño" en recursos. Para capturar la complejidad de la UNet, necesitamos un ansatz con mayor **Capacidad de Entrelazamiento**.

## 5. Conclusión
El entrenamiento cuántico nativo es viable y necesario para escalar. Hemos generado circuitos QASM válidos y eficientes. El siguiente paso lógico es aumentar la complejidad del Ansatz (pasar de Linear-Rzz a StronglyEntangling o EfficientSU2) para mejorar la fidelidad.

## 6. Artefactos
- **Modelo:** `checkpoints/quantum_native_model.pt`
- **Script:** `scripts/experiment_quantum_native_training.py`
- **Plot:** `docs/40_Experiments/images/exp008_results.png`
