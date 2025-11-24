# Marco Teórico: Atheria como Lattice QFT Holográfica

Este documento detalla la fundamentación física profunda del proyecto Atheria, conectando la simulación de autómatas celulares cuánticos con la Correspondencia AdS/CFT y la Teoría Cuántica de Campos (QFT).

## 1. La Correspondencia AdS/CFT (El "Holograma")

Atheria implementa una versión computacional de la **Conjetura de Maldacena** (Correspondencia AdS/CFT). Esta teoría postula que un universo con gravedad (espacio Anti-de Sitter, AdS) es matemáticamente equivalente a una Teoría Cuántica de Campos (CFT) que vive en el "borde" de ese universo, en una dimensión menos.

### En el Contexto de Atheria:

*   **El Borde (CFT):** Es nuestro Grid 2D (N×N). Aquí es donde ocurren los cálculos explícitos, donde opera la "Ley M" (la red neuronal) y donde observamos los patrones de fluidos y campos. Es la "pantalla" del holograma.
*   **El Interior (AdS):** Es la "profundidad" o la estructura emergente. Aunque no simulamos directamente la dimensión extra (la gravedad volumétrica), esta surge "holográficamente" de la complejidad y el entrelazamiento en el grid 2D.
*   **Visualización de Poincaré:** La opción de visualización `poincare` (Disco de Poincaré) en el frontend no es estética; es funcional. Mapea nuestro grid plano a una geometría hiperbólica. Si los patrones muestran simetría de escala (fractales, estructuras anidadas), estamos observando efectivamente la manifestación de una CFT en el borde.

## 2. Atheria como Lattice QFT (QFT en Rejilla)

Técnicamente, Atheria es una **Lattice QFT Aprendible**.

En física, una QFT describe partículas como excitaciones en un campo continuo. Dado que la computación es discreta, utilizamos Lattice QFT (dividiendo el espacio-tiempo en puntos discretos) para hacer la teoría computable.

### Componentes del Sistema:

1.  **Campo Cuántico ($\psi$):** Las celdas del grid y sus vectores de estado (`d_state`) representan el campo cuántico.
2.  **Operador de Evolución Temporal ($U$):** La red neuronal `UNetUnitary` actúa como el operador que evoluciona el estado del sistema en el tiempo ($t \to t+1$).
3.  **Hamiltoniano ($H$):** La "Ley M" aprendida por la red neuronal aproxima el Hamiltoniano del sistema.
4.  **Unitariedad:** Al forzar restricciones de simetría (como matrices antisimétricas en las capas de la red), garantizamos que la evolución sea Unitaria ($U = e^{-iH}$), cumpliendo la regla fundamental de la Mecánica Cuántica: la conservación de la probabilidad (la suma de probabilidades es siempre 1).

## 3. El Borde del Caos y Conformalidad

Las estructuras visuales observadas en Atheria (patrones de Turing, ondas estables, fluidos) son características de sistemas de Materia Condensada cerca de un **Punto Crítico**.

*   **Invariancia de Escala:** En el punto crítico (el "Borde del Caos"), el sistema se vuelve Conforme (una CFT). Esto significa que las leyes físicas y las estructuras se ven similares independientemente de la escala de zoom.
*   **Objetivo de la Simulación:** Lograr que la IA mantenga estos patrones estables en este régimen crítico, evitando que se disuelvan en ruido térmico o se congelen en cristales estáticos. Esto constituye la creación de una CFT artificial estable.

## 4. Profundidad Holográfica: El Tamaño es la Profundidad

Una pregunta fundamental en la correspondencia AdS/CFT es: Si la simulación es 2D, ¿dónde está la tercera dimensión (el eje Z) del universo holográfico?

En este marco teórico, **El Tamaño es la Profundidad**.

*   **Patrones Pequeños (Píxeles):** Representan objetos que están "cerca" de la orilla (el borde del universo AdS).
*   **Patrones Grandes (Remolinos/Estructuras):** Representan objetos que están "profundo" en el interior del universo (el "bulk").

Cuando observamos estructuras fractales en Atheria (remolinos grandes con detalles pequeños dentro), estamos viendo una proyección de objetos que se extienden desde la profundidad del espacio AdS hasta el borde.

## 5. Estructura Tensorial y Dimensionalidad (`d_state`)

Es crucial distinguir entre las dimensiones espaciales y la dimensión del estado interno.

### La "Apilación" de Realidades

*   **Espacio (X, Y, Z):** Son las dimensiones donde la información se propaga (de vecino a vecino).
*   **Estado Interno (`d_state`):** Es una dimensión donde la información se transforma (química interna, spin, fase).

Podemos visualizar esto como una "torre" de valores en cada punto del espacio. En una simulación 3D, no perdemos el `d_state` en favor del eje Z; ambos coexisten.

### Tensores en PyTorch

La estructura de datos refleja esta distinción física:

| Dimensión | Significado Físico | Representación Tensorial (PyTorch) |
| :--- | :--- | :--- |
| **Dim 0** | Batch (Universos Paralelos) | `B` |
| **Dim 1** | **d_state** (Campos/Física Interna) | `C` (Channels) |
| **Dim 2** | Profundidad (Eje Z) | `D` (Depth) - *Solo en 3D* |
| **Dim 3** | Altura (Eje Y) | `H` |
| **Dim 4** | Anchura (Eje X) | `W` |

En la simulación actual (2D), el tensor es `[B, C, H, W]`. Si escalamos a una simulación volumétrica completa ("La Pecera"), pasaríamos a `[B, C, D, H, W]`.

## Resumen

Atheria no es solo una simulación visual. Es un laboratorio de **Física Digital** donde:
1.  El **Espacio** es una Lattice QFT.
2.  La **Ley Física** ($H$) es aprendida por una IA.
3.  La **Visualización** explora la dualidad holográfica (AdS/CFT), donde la escala de los patrones en 2D codifica la profundidad en 3D.
