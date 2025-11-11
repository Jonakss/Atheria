# Arquitecturas de Modelos (Ley M)

Este documento describe las diferentes arquitecturas de redes neuronales disponibles en Aetheria para actuar como la "Ley M", la ley física fundamental que gobierna la evolución del universo QCA.

Todos los modelos se encuentran en el directorio `src/models/`.

---

### 1. U-Net (`unet.py`)

-   **Concepto**: Una arquitectura de red neuronal convolucional estándar con forma de "U", famosa por su eficacia en tareas de segmentación de imágenes. Consta de un camino de codificación (downsampling) para capturar contexto y un camino de decodificación (upsampling) para reconstrucción precisa.
-   **Física que Aprende**: Aprende a generar un "vector de cambio" (`delta_psi`) para cada celda. La evolución se calcula como `psi_t+1 = psi_t + delta_psi`.
-   **Características**:
    -   **No Conservativa**: No hay garantía inherente de que la "energía" (norma del vector de estado) se conserve. El modelo debe aprender la conservación a partir de la función de recompensa.
    -   **Flexible**: Puede aprender una amplia gama de físicas, incluyendo dinámicas disipativas o explosivas si no se penalizan.
-   **Parámetros Clave**:
    -   `d_state`: La dimensión del vector de estado de entrada/salida.
    -   `hidden_channels`: El número de canales en la primera capa convolucional, que determina el "ancho" o capacidad del modelo.
-   **Caso de Uso**: Ideal para experimentación general y cuando no se requiere una conservación estricta de la energía por diseño.

---

### 2. U-Net Unitaria (`unet_unitary.py`)

-   **Concepto**: Una modificación de la arquitectura U-Net diseñada para garantizar la conservación de la energía.
-   **Física que Aprende**: En lugar de aprender `delta_psi` directamente, aprende una **matriz antisimétrica `A`** para cada celda. La evolución se calcula como `delta_psi = A * psi`, lo que matemáticamente garantiza que la norma de `psi` se conserve a lo largo del tiempo (evolución unitaria).
-   **Características**:
    -   **Conservativa por Diseño**: La energía total del sistema se mantiene constante, previniendo explosiones o decaimientos numéricos. Esto estabiliza enormemente el entrenamiento.
    -   **Física de Rotación**: La matriz `A` actúa como un generador de rotaciones en el espacio de estados de alta dimensión. El modelo aprende a "rotar" los vectores de estado de las celdas en lugar de simplemente sumarles un delta.
-   **Parámetros Clave**:
    -   `d_state`: La dimensión del vector de estado. La salida del modelo será de `d_state * d_state` para representar la matriz `A`.
    -   `hidden_channels`: Similar a la U-Net estándar.
-   **Caso de Uso**: Es el modelo **recomendado y más avanzado**. Ideal para buscar complejidad emergente en sistemas estables y conservativos, simulando una física más "realista".

---

### 3. MLP (Perceptrón Multicapa) (`mlp.py`)

-   **Concepto**: Un modelo de red neuronal "fully-connected" simple. Procesa cada celda de forma independiente sin tener en cuenta a sus vecinos (no es convolucional).
-   **Física que Aprende**: Aprende una transformación simple para cada celda individualmente.
-   **Características**:
    -   **Local**: No puede aprender interacciones espaciales complejas, ya que no tiene "visión" de las celdas vecinas.
    -   **Rápido y Simple**: Útil para pruebas de baseline y para verificar que el pipeline de entrenamiento funciona correctamente.
-   **Caso de Uso**: Principalmente para depuración y como punto de referencia simple. No se espera que genere complejidad emergente significativa.

---

### Otros Modelos (Experimentales)

-   **`deep_qca.py`**: Una implementación de un modelo convolucional más profundo.
-   **`snn_unet.py`**: Un experimento con una U-Net que incorpora características de Redes Neuronales de Spiking (SNN), que podría ser explorado en el futuro para dinámicas temporalmente más complejas.
