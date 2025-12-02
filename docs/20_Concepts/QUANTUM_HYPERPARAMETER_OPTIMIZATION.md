# Optimización de Hiperparámetros Cuántica (Quantum Hyperparameter Optimization)

## Visión General
Utilizamos procesadores cuánticos (QPU) no para simular todo el universo, sino como "Oráculos" o "Maestros" que guían la búsqueda de la física perfecta. Esta estrategia aprovecha la capacidad de los algoritmos cuánticos para explorar espacios de búsqueda complejos y rugosos mejor que los métodos clásicos.

## 1. El Concepto: "El Buscador del Multiverso"
En lugar de probar configuraciones físicas una por una (fuerza bruta clásica), ponemos el chip cuántico en superposición para evaluar múltiples caminos simultáneamente.
* **Algoritmos:** VQE (Variational Quantum Eigensolver) y QAOA (Quantum Approximate Optimization Algorithm).
* **Función de Costo:** Minimizar la "Falta de Complejidad" (o maximizar la Entropía/Estabilidad).

## 2. Arquitectura Híbrida (The Hybrid Loop)
El sistema funciona en un bucle cerrado entre la QPU (IBM/IonQ) y la GPU clásica.

1.  **QPU (Propuesta):** El chip cuántico, ejecutando un circuito variacional (Ansatz), propone un conjunto de parámetros ($\theta$) que se mapean a hiperparámetros físicos (ej. `GAMMA_DECAY`, `LR_RATE`).
2.  **GPU (Evaluación):** El simulador clásico corre una versión rápida de Aetheria (ej. 50 pasos) con esos parámetros.
3.  **Medición (Reward):** Se calcula una métrica de "Vida" (ej. Entropía de Shannon + Estabilidad Temporal).
4.  **Feedback (Ajuste):** El valor del Reward se devuelve al optimizador cuántico (SPSA), que ajusta los ángulos del circuito para la siguiente iteración.

## 3. Implementación con Qiskit Runtime
Utilizamos `EstimatorV2` de Qiskit Runtime para ejecutar este bucle eficientemente en la nube de IBM.
* **Circuito:** Un circuito parametrizado simple donde cada rotación controla un hiperparámetro.
* **Optimizador:** SPSA (Simultaneous Perturbation Stochastic Approximation), ideal para entornos ruidosos.
* **Sesiones:** Uso de `Session` para minimizar la latencia de cola entre iteraciones.

## 4. Ventajas
* **Exploración Global:** Evita mínimos locales donde los optimizadores clásicos (descenso de gradiente) suelen atascarse.
* **Eficiencia de Muestreo:** Puede encontrar "Islas de Estabilidad" en el espacio de parámetros con menos evaluaciones totales.
