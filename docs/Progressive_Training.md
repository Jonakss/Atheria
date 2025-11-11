# Guía de Entrenamiento Progresivo (Aprendizaje Curricular)

El Entrenamiento Progresivo, también conocido como Aprendizaje Curricular, no es una configuración automática, sino una **estrategia manual de varios pasos** para entrenar modelos complejos de forma más eficiente. La idea es "transferir" el conocimiento de un modelo entrenado en una tarea simple a un nuevo modelo que debe aprender una tarea más compleja.

Esto evita que el modelo tenga que aprender todo desde cero y guía el proceso de aprendizaje, resultando en una convergencia más rápida y estable.

## Cómo Funciona en Aetheria

La funcionalidad se habilita a través de una variable en `src/config.py`:

-   **`LOAD_FROM_EXPERIMENT`**:
    -   Si se establece con el nombre de un experimento anterior (ej. `"Fase1_Estabilidad_4D"`), el trainer cargará los pesos guardados de ese experimento al iniciar un **nuevo** entrenamiento.
    -   Si se deja como `None`, el entrenamiento comenzará desde cero con pesos inicializados aleatoriamente.

El sistema utiliza `torch.load_state_dict(..., strict=False)`, lo que permite cargar pesos incluso si las arquitecturas no son idénticas (ej. cambiar `D_STATE` o `HIDDEN_CHANNELS`). Las capas que coincidan en forma y nombre se cargarán, y las que no, se dejarán con su inicialización aleatoria.

## La "Receta": Un Ejemplo de Flujo de Trabajo

A continuación se describe un flujo de trabajo de 3 fases para entrenar un modelo complejo que exhiba una "cosmología" (un vacío estable con "materia" emergente).

---

### Fase 1: Entrenar la Estabilidad (El Vacío)

**Objetivo**: Enseñar al modelo a crear un "vacío" estable. La tarea más fácil.

**`src/config.py`:**
```python
EXPERIMENT_NAME = "Fase1_Estabilidad_4D"
CONTINUE_TRAINING = False
LOAD_FROM_EXPERIMENT = None  # Empezar desde cero

STATE_VECTOR_DIM = 4         # Dimensión baja, tarea simple
HIDDEN_CHANNELS = 32         # Modelo pequeño, aprende más rápido
LR_RATE_M = 1e-4             # Tasa de aprendizaje más alta

# --- Recompensas ---
PESO_QUIETUD = 10.0          # ¡Muy alta! Solo nos importa la estabilidad.
PESO_COMPLEJIDAD_LOCALIZADA = 0.0 # ¡Apagada! No queremos materia todavía.
```
**Acción**: Inicia el entrenamiento desde la UI.
**Resultado**: Un modelo en `output/experiments/Fase1_Estabilidad_4D/` que es muy bueno para no hacer nada (un universo vacío y estable).

---

### Fase 2: Entrenar la Complejidad (La Materia)

**Objetivo**: Cargar el "vacío" de la Fase 1 y enseñarle a crear "materia" dentro de él.

**`src/config.py`:**
```python
EXPERIMENT_NAME = "Fase2_Cosmologia_4D" # ¡Nuevo nombre!
CONTINUE_TRAINING = False
LOAD_FROM_EXPERIMENT = "Fase1_Estabilidad_4D" # ¡Aquí está la magia!

STATE_VECTOR_DIM = 4         # Mantenemos la dimensión
HIDDEN_CHANNELS = 64         # Aumentamos la capacidad del modelo
LR_RATE_M = 1e-5             # Bajamos la tasa para un ajuste fino

# --- Recompensas ---
PESO_QUIETUD = 1.0           # Equilibrado
PESO_COMPLEJIDAD_LOCALIZADA = 20.0 # ¡Encendida! Ahora queremos complejidad.
```
**Acción**: Inicia el entrenamiento desde la UI.
**Resultado**: El trainer carga los pesos de la Fase 1 y comienza a entrenar la nueva física "cosmológica" desde el Episodio 0. El modelo aprende mucho más rápido porque ya sabe cómo mantener la estabilidad del vacío.

---

### Fase 3: Escalar la Física (Aumentar Dimensiones)

**Objetivo**: Cargar el modelo 4D y expandir su "física" a un espacio de estados de mayor dimensión (8D).

**`src/config.py`:**
```python
EXPERIMENT_NAME = "Fase3_Cosmologia_8D" # ¡Nuevo nombre!
CONTINUE_TRAINING = False
LOAD_FROM_EXPERIMENT = "Fase2_Cosmologia_4D" # Cargamos el modelo anterior

STATE_VECTOR_DIM = 8         # ¡Aumentamos la dimensión!
HIDDEN_CHANNELS = 64         # Mantenemos la capacidad
LR_RATE_M = 1e-6             # Aún más bajo para un ajuste muy fino

# --- Recompensas ---
PESO_QUIETUD = 1.0
PESO_COMPLEJIDAD_LOCALIZADA = 20.0
```
**Acción**: Inicia el entrenamiento desde la UI.
**Resultado**: El trainer carga los pesos del modelo 4D en el nuevo modelo 8D. `strict=False` se encarga de que las capas convolucionales internas (que no cambian) se carguen, mientras que las capas de entrada y salida (que ahora son más grandes debido a `D_STATE=8`) se inicializan aleatoriamente. El modelo converge a una física de 8D mucho más rápido que si empezara desde cero.
