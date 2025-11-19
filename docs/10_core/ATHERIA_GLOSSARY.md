Glosario Técnico de Atheria

Este documento define los términos específicos usados en el código y la arquitectura.

Conceptos Fundamentales

Ley M (M-Law): El modelo de red neuronal (U-Net, SNN) que gobierna la evolución del estado t a t+1. Es la "función de onda" del universo.

d_state (Dimensiones de Estado): El número de canales por celda (ej. 16). Representa la magnitud de los diferentes Campos Cuánticos en ese punto del espacio.

Sparse Engine (Motor Disperso): El motor de simulación de Atheria 4. Solo almacena en RAM las celdas con energía > $\epsilon$.

Harmonic Vacuum (Vacío Armónico): Sistema que reemplaza el valor 0 o random() en las zonas vacías. Genera un campo de ondas estacionarias deterministas ($\sum \sin(kx-\omega t)$) para que la materia tenga un "terreno" sobre el que interactuar.

Tipos de Ruido (The Noise Stack)

IonQ Noise (Hostil): Ruido de Bit-Flip y Phase-Flip aplicado durante el entrenamiento. Su objetivo es destruir estructuras débiles para forzar la evolución.

QED Noise (Ambiental): Fluctuaciones de punto cero generadas por el Vacío Armónico. Permite el movimiento y evita el estancamiento.

Estructuras de Datos

Chunk: Un bloque de espacio de 32x32x32 celdas. Unidad mínima de procesamiento en el motor disperso.

Active Matter: Coordenadas (x, y, z) que contienen excitaciones de campo significativas.

Viewport: La "Cámara". Una región densa muestreada temporalmente del universo infinito para ser enviada al frontend.

Las 5 Épocas (Epochs)

Era Cuántica: Sopa de probabilidad y ruido.

Era de Partículas: Formación de cristales simétricos estables.

Era Química: Interacción dinámica y movimiento (Gliders).

Era Gravitacional: Agregación de materia en grandes estructuras (Planetas).

Era Biológica: Homeostasis y replicación de información.