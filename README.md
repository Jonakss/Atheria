# AETHERIA: Simulación de Complejidad Emergente con QCA

Bienvenido a AETHERIA, una aplicación para simular la emergencia de estructuras complejas a partir de reglas físicas fundamentales.

Este proyecto modela un universo discreto como una cuadrícula de Autómatas Celulares Cuánticos (QCA). La evolución de este universo no está pre-programada, sino que es gobernada por una **"Ley M" (Ley Fundamental)**: un modelo de Deep Learning (como un MLP o una U-Net) que se entrena desde cero.

El objetivo es descubrir, mediante un proceso de "evolución artificial" (Aprendizaje por Refuerzo), una Ley M que opere en el **"Borde del Caos"**: el régimen crítico donde la información puede propagarse, la estabilidad se mantiene y la complejidad estructural emerge espontáneamente.

Esta aplicación está construida como una **App de Lightning AI**, permitiendo un entrenamiento pesado en GPU y un despliegue de simulación en tiempo real a través de un servidor WebSocket.

-----

## 🚀 Arquitectura del Proyecto

El proyecto está separado en un lanzador de aplicación (`app.py`), una interfaz de usuario (`ui.py`), un lanzador de script local (`main.py`) y un paquete de código fuente (`src/`).

```
aetheria/
├── app.py              <-- 🚀 El lanzador de la App Lightning (Frontend + Backend)
├── ui.py               <-- 🖥️ El visor web (Streamlit UI)
├── main.py             <-- 🔬 El lanzador para ejecución local (entrenamiento/scripts)
├── requirements.txt    <-- 📋 Dependencias del proyecto
│
├── src/                <-- 🧠 Todo el código fuente
│   ├── __init__.py
│   ├── config.py         <-- ⚙️ ¡Parámetros globales y flags de ejecución aquí!
│   │
│   ├── qca_engine.py     <-- 🌌 Clases Aetheria_Motor y QCA_State
│   ├── qca_operator_mlp.py  <-- 🧬 Ley M v1: MLP 1x1 (Visión local, "míope")
│   ├── qca_operator_unet.py <-- 🧬 Ley M v2: U-Net (Visión regional, "consciente")
│   │
│   ├── trainer.py        <-- 🏋️ Clase de entrenamiento (QC_Trainer_v3)
│   ├── visualization.py  <-- 🎨 Funciones get_frame_gpu()
│   ├── utils.py          <-- 📦 Funciones de ayuda (load/save_state)
│   │
│   ├── pipeline_train.py   <-- 🏭 Script: FASE 5 (Entrenamiento)
│   ├── pipeline_viz.py     <-- 🎬 Script: FASE 6 (Generar Vídeos)
│   └── pipeline_server.py  <-- 📡 Script: FASE 7 (Servidor WebSocket)
│
└── output/             <-- 📊 Todos los resultados (vídeos y checkpoints)
    ├── training_checkpoints/
    └── simulation_checkpoints/
```

-----

## ⚙️ Cómo Empezar

### 1\. Instalación

Asegúrate de tener todas las dependencias instaladas en tu entorno.

```bash
pip install -r requirements.txt
```

### 2\. Configuración

**Casi todo se controla desde `src/config.py`**. Antes de ejecutar, revisa este archivo para:

  * Ajustar los *flags* de ejecución (`RUN_TRAINING`, `RUN_LARGE_SIM`, etc.).
  * Configurar los parámetros de entrenamiento (`GRID_SIZE_TRAINING`, `EPISODES_TO_ADD`).
  * Configurar los parámetros de la simulación (`GRID_SIZE_INFERENCE`).

### 3\. Elegir tu "Ley M" (El Cerebro)

Puedes cambiar fácilmente qué modelo de física quieres entrenar o ejecutar. Abre `src/pipeline_train.py` y `src/pipeline_server.py` y edita el "Selector de Modelo" en la parte superior:

```python
# --- Elige tu "Ley M" (Cerebro) aquí ---

# Opción 1: El MLP 1x1 original (Rápido, pero "míope")
from .qca_operator_mlp import QCA_Operator_MLP as ActiveModel

# Opción 2: La U-Net (Más lenta, pero con "conciencia regional")
# from .qca_operator_unet import QCA_Operator_UNet as ActiveModel
```

-----

## 🏃 Cómo Ejecutar

Este proyecto tiene **dos modos de ejecución principales**:

### Modo 1: Entrenamiento y Scripting (Local)

Usa `main.py` para tareas de "un solo uso" como entrenar un nuevo modelo o generar un lote de videos.

1.  **Configura:** Abre `src/config.py` y pon:
      * `RUN_TRAINING = True`
      * `RUN_POST_TRAINING_VIZ = True`
      * `RUN_LARGE_SIM = False` (¡Importante\!)
2.  **Ejecuta:**
    ```bash
    python main.py
    ```
3.  **Resultado:** El script ejecutará el `pipeline_train.py` y luego el `pipeline_viz.py`. Todos los modelos (`.pth`) y videos (`.mp4`) se guardarán en la carpeta `output/`.

### Modo 2: Servidor de Simulación (Producción)

Usa `app.py` para lanzar la simulación persistente como un servicio en la nube (o localmente) con un visor web en tiempo real.

1.  **Configura:** Abre `src/config.py` y pon:
      * `RUN_TRAINING = False`
      * `RUN_POST_TRAINING_VIZ = False`
      * `RUN_LARGE_SIM = True`
2.  **Ejecuta (Localmente):**
    ```bash
    lightning run app app.py
    ```
3.  **Ejecuta (En la Nube de Lightning AI):**
    ```bash
    lightning run app app.py --cloud
    ```
4.  **Resultado:** Esto lanzará el backend de simulación (`SimulationServer`) en una GPU y el frontend (`ui.py`) en un servidor web. Abre la URL que te da la terminal para ver la simulación en tiempo real.

-----

## 📊 Interpretación de Resultados

El visor te permite observar la dinámica emergente del QCA en tiempo real:

  * **Densidad:** Mapa de calor que muestra la concentración de "energía" o "materia".
  * **Canales Internos:** Mapeo a RGB de los primeros canales del estado. Ayuda a ver la actividad de los componentes del campo.
  * **Magnitud de Estado:** Intensidad total del vector de estado en cada celda.
  * **Fase de Estado:** Coherencia de fase, crucial para el comportamiento tipo onda.
  * **Cambio de Estado / Actividad:** Resalta las regiones activas o "vivas" del universo.

-----

## 💾 Checkpointing y Reanudación

El proyecto guarda el progreso automáticamente en la carpeta `output/`.

  * **Checkpoints de Entrenamiento (`output/training_checkpoints/`):**
      * Contienen el estado del modelo, optimizador e historial.
      * Para reanudar el entrenamiento, pon `CONTINUE_TRAINING = True` en `src/config.py`.
  * **Checkpoints de Simulación (`output/simulation_checkpoints/`):**
      * Contienen el estado crudo (`x_real`, `x_imag`) de la simulación grande.
      * Para reanudar una simulación, pon `LOAD_STATE_CHECKPOINT_INFERENCE = True` en `src/config.py`.

-----

## 🧬 Parámetros Clave en `src/config.py`

### Arquitectura y Entrenamiento

  * `GRID_SIZE_TRAINING`: Tamaño de la cuadrícula para entrenar (ej. 256).
  * `D_STATE`: Canales/dimensiones de cada celda (ej. 21).
  * `HIDDEN_CHANNELS`: Ancho de la Ley M (ej. 64 para U-Net, 256 para MLP).
  * `EPISODES_TO_ADD`: Cuántos episodios de entrenamiento ejecutar.
  * `PERSISTENCE_COUNT`: Pasos de BPTT (memoria del entrenamiento).

### Recompensas (El "Objetivo" de la Física)

  * `ALPHA_START`/`ALPHA_END`: Peso de la recompensa de **complejidad** (`R_Density_Target`).
  * `GAMMA_START`/`GAMMA_END`: Peso de la recompensa de **estabilidad** (`R_Stability`).
  * `BETA_CAUSALITY`: Peso de la recompensa de **actividad** (`R_Causality`).
  * `LAMBDA_ACTIVITY_VAR`: Recompensa por varianza de actividad (crea "vida" interesante).
  * `LAMBDA_VELOCIDAD`: Recompensa por la varianza de la densidad (crea "movimiento").

### Simulación y Servidor

  * `GRID_SIZE_INFERENCE`: Tamaño de la cuadrícula de producción (ej. 468, 1024).
  * `REAL_TIME_VIZ_INTERVAL`: Cada cuántos pasos se envía un frame al visor (ej. 5).
  * `REAL_TIME_VIZ_TYPE`: Qué tipo de frame enviar (`density`, `change`, `phase`, etc.).
  * `REAL_TIME_VIZ_DOWNSCALE`: Factor de reducción de la imagen para el visor (ej. 2).