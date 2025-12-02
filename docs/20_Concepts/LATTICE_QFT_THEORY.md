# Teoría Cuántica de Campos en Retículo (Lattice QFT) en Aetheria

## Introducción
Aetheria no es una simulación de fluidos convencional; es una implementación aprendible de una **Lattice Quantum Field Theory (QFT)**. Este documento establece el marco teórico que conecta nuestro código con la física fundamental.

## 1. El Retículo (Lattice)
En física, una QFT describe campos continuos. Para simularlos en computadoras, discretizamos el espacio-tiempo en una rejilla o retículo.
* **Celdas:** Cada punto $(x, y)$ del grid contiene un vector de estado complejo $\psi \in \mathbb{C}^d$.
* **Tensor:** Representado como `[Batch, d_state, H, W]`.

## 2. La Ley M como Operador de Evolución
La evolución temporal del sistema está gobernada por un operador unitario $U = e^{-iHt}$.
* **En Física:** $H$ es el Hamiltoniano (energía total).
* **En Aetheria:** $U$ es aproximado por una Red Neuronal (U-Net).
* **Restricción Unitaria:** Para que la simulación sea física, la red debe preservar la norma del estado ($||\psi||^2 = 1$). Esto garantiza la conservación de la probabilidad.

## 3. Holografía y AdS/CFT
Interpretamos nuestra simulación 2D bajo el Principio Holográfico.
* **Frontera (CFT):** El grid 2D donde corren los cálculos es la "superficie" del universo.
* **Volumen (AdS):** La complejidad emergente y los patrones de escala sugieren una dimensión extra implícita ($Z$).
* **Visualización:** El mapeo al disco de Poincaré nos permite visualizar esta geometría hiperbólica subyacente.

## 4. El Límite de Harlow
Basado en la teoría de Daniel Harlow (MIT):
* Un universo cerrado con evolución unitaria perfecta puede tener un estado global simple (casi estático).
* La complejidad que observamos ("vida", "caos") es producto del *coarse-graining* (baja resolución) del observador interno.
* **Experimento:** Entrenar con `GAMMA_DECAY=0` y medir la divergencia entre la Fidelidad matemática (alta) y la Entropía visual (alta).
