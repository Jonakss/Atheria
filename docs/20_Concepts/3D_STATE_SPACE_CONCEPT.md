---
id: concepto_espacio_estados_3d
tipo: concepto_fisico
tags: [arquitectura, 3d, tensores, fisica]
---

# Conceptualización del Espacio de Estados en 3D

Este documento formaliza la estructura de datos y la interpretación física para la evolución del motor de Aetheria de 2D a 3D.

## 1. De la Superficie al Volumen

Para entender cómo escala nuestra simulación, utilizamos la analogía de la "pila de dimensiones".

### El Modelo Actual (2D - "La Superficie")
En la simulación 2D, nuestro universo es una membrana plana $N \times N$.
* **Espacio ($X, Y$):** Una cuadrícula donde las células tienen posición.
* **Estado (`d_state`):** En cada punto $(x, y)$ de la cuadrícula, existe un vector de propiedades.
    * *Analogía Visual:* Imagina un tablero de ajedrez donde en cada casilla hay una **torre de fichas de lego** de diferentes colores. La altura de esa torre es la dimensión del `d_state`.
* **Tensor:** `[Batch, d_state, Altura, Anchura]`

### El Nuevo Modelo (3D - "El Tanque")
Al pasar a 3D, no convertimos el `d_state` en la tercera dimensión espacial. En su lugar, "apilamos" infinitas superficies 2D una sobre otra para crear volumen.
* **Espacio ($X, Y, Z$):** Ahora el universo es un cubo o "pecera". Las células tienen profundidad.
* **Estado (`d_state`):** La "torre de legos" (el vector de propiedades) sigue existiendo **dentro** de cada punto $(x, y, z)$ del cubo.
* **Tensor:** `[Batch, d_state, Profundidad, Altura, Anchura]`

## 2. Representación Matemática (Tensores)

Para la implementación en PyTorch (usando `Conv3d`), la estructura de datos cambia de 4D a 5D.

| Dimensión | Nombre en Código | Significado Físico |
| :--- | :--- | :--- |
| **0** | `Batch (B)` | Universos paralelos simulados simultáneamente. |
| **1** | `Channels (C)` | **`d_state`**: La química interna. Es el vector que se transforma pero no se mueve espacialmente. |
| **2** | `Depth (D)` | **Eje Z**: La nueva dimensión espacial. Define "dónde" estás en la profundidad del tanque. |
| **3** | `Height (H)` | **Eje Y**: Posición vertical. |
| **4** | `Width (W)` | **Eje X**: Posición horizontal. |

> **Nota:** Matemáticamente, el sistema opera en una "Hiper-Matriz" de 4 dimensiones (3 espaciales + 1 de estado), aunque computacionalmente lo representamos como un tensor 5D.

## 3. Diferencia entre Dimensiones Espaciales y de Estado

Es crucial distinguir entre moverse y transformarse:

1.  **Dimensiones Espaciales ($X, Y, Z$):**
    * Son los ejes de **propagación**.
    * La información viaja a través de ellos mediante las operaciones de convolución (`Conv3d`).
    * Una célula en $(x,y,z)$ influye en sus vecinos inmediatos en 3D (arriba/abajo, norte/sur, este/oeste).

2.  **Dimensión de Estado (`d_state`):**
    * Es el eje de **transformación**.
    * La información se mezcla dentro de este vector (mediante multiplicaciones de matrices densas en la red neuronal).
    * Representa la complejidad local: masa, energía, fase, tipo de partícula, etc.

## 4. Visualización Holográfica (AdS/CFT)

Dado que visualizar un cubo denso de datos es complejo, mantenemos la conexión con el principio holográfico:

* **La Simulación:** Ocurre en el "Bulk" 3D (el interior del tanque).
* **La Observación:** Podemos proyectar cortes o sombras de este volumen en una superficie 2D.
* **Hipótesis:** Si la física es correcta, la proyección 2D de nuestro tanque 3D debería conservar patrones coherentes, validando la correspondencia entre el volumen y su superficie.
