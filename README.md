# AETHERIA: Simulación de Fenómenos Cuánticos Emergentes mediante QCA-Deep.

Este cuaderno interactivo de Colab/Kaggle te permite explorar el proyecto AETHERIA, un esfuerzo por simular la emergencia de la estructura y la complejidad dinámica en un universo discreto basado en Autómatas Celulares Cuánticos Profundos (QCA-Deep). Inspirado en la Teoría Cuántica de Campos (TCC), modelamos el universo como una grilla de campos cuánticos que evolucionan siguiendo una "Ley M" aprendida mediante Optimización de Hiperparámetros (HpO) para encontrar el "Borde del Caos", un régimen de Acoplamiento Crítico donde la complejidad es máxima y la estabilidad se mantiene.

## Estructura del Cuaderno

El cuaderno está organizado en fases lógicas para guiarte a través del pipeline de AETHERIA:

*   **FASE 0: SETUP E IMPORTACIONES:** Configuración inicial del entorno, detección de dispositivo (CPU/GPU) e importación de librerías necesarias.
*   **FASE 1: CLASES DEL MOTOR QCA:** Definición de las clases fundamentales: `QCA_State` (el estado del universo), `QCA_Operator_Deep` (la Ley M entrenable) y `Aetheria_Motor` (que gestiona la evolución).
*   **FASE 3: HIPERPARÁMETROS OPTIMIZADOS Y NUEVAS MÉTRICAS:** Definición de todos los parámetros configurables para el entrenamiento y la simulación, incluyendo los pesos para las nuevas recompensas (`R_Activity_Var`, `R_Velocidad`).
*   **FASE 4: FUNCIONES DE VISUALIZACIÓN Y CHECKPOINTING DE ESTADO:** Funciones auxiliares para generar frames visuales y guardar/cargar el estado crudo del QCA.
*   **FASE 5: LÓGICA PRINCIPAL DE ENTRENAMIENTO:** Inicialización y ejecución del bucle de entrenamiento PEF v3, incluyendo carga/guardado de checkpoints y lógica de reactivación.
*   **Fase 6: VISUALIZACIÓN POST-ENTRENAMIENTO (Tamaño de Entrenamiento):** Ejecución de una simulación corta con el modelo entrenado en el tamaño de entrenamiento y generación de videos de visualización.
*   **FASE 7: LÓGICA PRINCIPAL DE SIMULACIÓN GRANDE PROLONGADA:** Configuración y ejecución de una simulación a gran escala (`1024x1024`) utilizando el modelo entrenado, con opciones para video optimizado, visualización en tiempo real y guardado periódico del estado crudo.

*(Nota: Las celdas pueden no seguir siempre un orden numérico estricto en el cuaderno, pero la lógica de ejecución sí respeta el flujo descrito.)*

## Cómo Ejecutar

1.  **Ejecutar Celdas Secuencialmente:** Ejecuta las celdas en el orden en que aparecen en el cuaderno, de arriba hacia abajo. Asegúrate de que cada celda termine su ejecución antes de pasar a la siguiente.
2.  **Configurar Parámetros:** La celda **FASE FINAL: PARÁMETROS GLOBALES Y EJECUCIÓN PRINCIPAL** contiene todos los parámetros configurables. Puedes modificar estos valores para experimentar con diferentes configuraciones de entrenamiento, simulación, guardado de video, visualización y checkpointing.
3.  **Controlar Fases:** En la celda de parámetros, ajusta las variables booleanas `RUN_TRAINING`, `RUN_POST_TRAINING_VIZ` y `RUN_LARGE_SIM` para controlar qué partes del pipeline deseas ejecutar.
4.  **Continuar Entrenamiento/Simulación:**
    *   Para reanudar el entrenamiento, establece `CONTINUE_TRAINING = True` en la celda de parámetros. El script buscará automáticamente el último checkpoint en `checkpoints_optimized/`.
    *   Para reanudar una simulación grande, establece `LOAD_STATE_CHECKPOINT_INFERENCE = True` y opcionalmente `STATE_CHECKPOINT_PATH_INFERENCE` en la celda de parámetros. El script buscará el último checkpoint en `large_sim_checkpoints_1024/` por defecto si no se especifica una ruta.

## Parámetros Clave

*   **`GRID_SIZE_TRAINING`, `D_STATE`, `HIDDEN_CHANNELS`:** Definen la arquitectura del modelo entrenado.
*   **`EPISODES_TO_ADD`, `STEPS_PER_EPISODE`, `LR_RATE_M`, `PERSISTENCE_COUNT`:** Controlan el proceso de entrenamiento.
*   **`ALPHA_START`, `ALPHA_END`, `GAMMA_START`, `GAMMA_END`, `BETA_CAUSALITY`, `LAMBDA_ACTIVITY_VAR`, `LAMBDA_VELOCIDAD`:** Pesos y configuración del annealing para las diferentes métricas de recompensa/penalización que guían el entrenamiento.
*   **`TARGET_STD_DENSITY`, `EXPLOSION_THRESHOLD`, `EXPLOSION_PENALTY_MULTIPLIER`:** Objetivos y umbrales para las penalizaciones de densidad y explosión.
*   **`STAGNATION_WINDOW`, `MIN_LOSS_IMPROVEMENT`, `REACTIVATION_COUNT`, `REACTIVATION_STATE_MODE`, `REACTIVATION_LR_MULTIPLIER`:** Configuración de la lógica de detección de estancamiento y reactivación.
*   **`GRADIENT_CLIP`:** Límite para el clipping de gradientes.
*   **`SAVE_EVERY_EPISODES`:** Frecuencia de guardado de checkpoints durante el entrenamiento.
*   **`NUM_FRAMES_VIZ`, `FPS_VIZ_TRAINING`:** Parámetros para la visualización post-entrenamiento.
*   **`GRID_SIZE_INFERENCE`, `NUM_INFERENCE_STEPS`:** Tamaño y duración de la simulación grande.
*   **`LARGE_SIM_CHECKPOINT_INTERVAL`:** Frecuencia de guardado del estado crudo durante la simulación grande.
*   **`VIDEO_FPS`, `VIDEO_SAVE_INTERVAL_STEPS`, `VIDEO_DOWNSCALE_FACTOR`, `VIDEO_QUALITY`:** Parámetros para optimizar el tamaño de los archivos de video generados durante la simulación grande.
*   **`REAL_TIME_VIZ_INTERVAL`, `REAL_TIME_VIZ_TYPE`, `REAL_TIME_VIZ_DOWNSCALE`:** Configuración para la visualización en tiempo real en el output del notebook.

## Interpretación de Resultados

La visualización te permite observar la dinámica emergente del QCA:

*   **Densidad:** Mapa de calor que muestra la suma de probabilidades al cuadrado por celda. Indica dónde se concentra la "energía" o "mater
ia".
*   **Canales Internos:** Visualización de los primeros canales del estado, mapeados a colores RGB. Ayuda a ver la actividad individual de los componentes del campo.
*   **Magnitud de Estado:** Escala de grises que representa la magnitud total del vector de estado en cada celda. Simboliza la "intensidad del campo" local.
*   **Fase de Estado:** Colores mapeados al tono (Hue) según la fase del vector de estado. Patrones de color coherentes indican coherencia de fase, crucial para el comportamiento tipo onda.
*   **Cambio de Estado / Actividad:** Escala de grises que muestra cuánto cambia el estado en cada celda por paso temporal. Resalta las regiones activas o de "disipación".

**Conexión con la Visualización 3D (Futuro Trabajo):**

Como se describe en el documento, la Amplitud del estado (`sqrt(x_real^2 + x_imag^2)`) se correlaciona conceptualmente con la Opacidad y el Brillo en una visualización 3D, mientras que la Fase (`atan2(x_imag, x_real)`) se mapea al Tono de Color. El eje Z representaría el Tiempo, permitiendo ver las "líneas de mundo" de los patrones emergentes propagándose a través del tiempo.

## Checkpointing y Reanudación

Durante el entrenamiento y la simulación grande, el script guarda periódicamente checkpoints en los directorios especificados (`checkpoints_optimized/` y `large_sim_checkpoints_1024/`).

*   **Checkpoints de Entrenamiento:** Contienen el estado del modelo, optimizador, historial y contadores. Permiten reanudar el entrenamiento desde el último punto guardado (`CONTINUE_TRAINING = True`). También se guarda el "mejor" modelo (`qca_best_eps*.pth`) basado en la menor pérdida.
*   **Checkpoints de Simulación Grande:** Contienen el estado crudo del QCA (`x_real`, `x_imag`) y el número de paso. Permiten reanudar simulaciones largas desde un punto específico (`LOAD_STATE_CHECKPOINT_INFERENCE = True`).

Estos archivos `.pth` son esenciales para reanudar corridas largas o para analizar el estado del universo simulado en puntos temporales específicos.

## Futuro Trabajo: Visualización 3D en Streaming

El guardado periódico del estado crudo en la simulación grande está diseñado para ser compatible con una futura visualización 3D en streaming. La idea es que una aplicación externa pueda cargar estos datos de estado (o recibirlos vía streaming) y renderizar el volumen espacio-temporal utilizando técnicas como Ray Marching en GPU, mapeando Amplitud a Opacidad/Brillo y Fase a Color. Esto permitiría una exploración interactiva y continua de la evolución del universo simulado.