# Proceso de Trabajo: Validación de la Arquitectura U-Net

Este plan describe los pasos para entrenar, validar y observar la nueva Ley M basada en U-Net.

## Paso 1: Verificación de Configuración (El "Humo Test")

El primer objetivo es asegurarnos de que el modelo pueda entrenar sin crashear por memoria.

1.  **Verificar `src/config.py`:**
    * `RUN_TRAINING = True`
    * `RUN_POST_TRAINING_VIZ = False`
    * `RUN_LARGE_SIM = False`
    * `HIDDEN_CHANNELS = 64` (o `32` si `64` falla).
    * `EPISODES_TO_ADD = 20` (Solo queremos un entrenamiento corto para probar).
    * `CONTINUE_TRAINING = False` (Para forzar un re-entrenamiento desde cero).

2.  **Ejecutar el Entrenamiento:**
    ```bash
    python main.py
    ```

3.  **Observar:** ¿El entrenamiento se completa sin errores de `std::bad_alloc` (RAM) o `CUDA out of memory` (VRAM)? Si es así, la arquitectura es viable.

## Paso 2: Entrenamiento Real (Línea de Base de U-Net)

Ahora que sabemos que funciona, vamos a entrenarla de verdad para que aprenda una física coherente.

1.  **Verificar `src/config.py`:**
    * `EPISODES_TO_ADD = 500` (O tu número de episodios estándar).
    * `CONTINUE_TRAINING = False` (Queremos un modelo U-Net limpio).

2.  **Ejecutar y Esperar:**
    ```bash
    python main.py
    ```
    *Esto tardará significativamente más que tu entrenamiento anterior.*

## Paso 3: Observación (El Servidor de Inferencia)

Una vez que el entrenamiento termine y tengas un nuevo archivo `.pth` en tu carpeta `output/training_checkpoints/`, es hora de ver qué aprendió.

1.  **Verificar `src/config.py`:**
    * `RUN_TRAINING = False`
    * `RUN_LARGE_SIM = True`
    * `LOAD_STATE_CHECKPOINT_INFERENCE = False` (Queremos empezar desde un estado de ruido nuevo).

2.  **Ejecutar el Servidor (Modo Producción):**
    ```bash
    lightning run app app.py
    ```

3.  **Abrir el Visor:** Abre la URL principal de la app (la que termina en `7501` o `8080`, *no* la que termina en `8765`).

4.  **Analizar:** Observa la simulación.
    * ¿Ves las estructuras a gran escala que esperábamos?
    * ¿Se siente más "organizado" que tu simulación anterior?
    * ¿Es estable o colapsa?

## Paso 4: Iteración y Siguiente Mejora

Aquí es donde te conviertes en científico.

1.  **Ajustar Recompensas:** ¿La U-Net no converge? Quizás necesita recompensas diferentes. Intenta ajustar los valores `ALPHA`, `GAMMA`, etc., en `src/config.py` y vuelve al Paso 2.

2.  **Próxima Arquitectura (Memoria Celular):** Si la U-Net funciona y te gusta, el siguiente paso lógico que discutimos es añadir **`ConvLSTM`** (memoria celular). Esto le permitiría a tus estructuras de U-Net "recordar" estados pasados, llevando a comportamientos aún más complejos.