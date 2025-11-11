# AETHERIA: Laboratorio de Complejidad Emergente

AETHERIA es un laboratorio de software para simular la emergencia de estructuras complejas a partir de "Leyes Físicas" aprendibles. El sistema utiliza Autómatas Celulares Cuánticos (QCA) cuya evolución es gobernada por un modelo de Deep Learning (la "Ley M").

El objetivo es descubrir, mediante entrenamiento, una Ley M que opere en el **"Borde del Caos"**: el régimen crítico donde la estabilidad se mantiene, la información se propaga y la complejidad estructural emerge espontáneamente.

El proyecto está construido como una aplicación unificada con un backend en Python (`aiohttp`, `torch`) y un frontend moderno en React (`Vite`, `Mantine`), permitiendo controlar todo el ciclo de vida de la experimentación (entrenamiento, simulación, análisis) desde una única interfaz web.

## 🚀 Cómo Empezar

### 1. Prerrequisitos

- Python 3.9+
- Node.js 18+ y npm

### 2. Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd Atheria
    ```

2.  **Configura el Backend:**
    Crea un entorno virtual e instala las dependencias de Python.
    ```bash
    python3 -m venv torch_venv
    source torch_venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configura el Frontend:**
    Navega al directorio del frontend e instala las dependencias de Node.js.
    ```bash
    cd frontend
    npm install
    ```

### 3. Ejecutar la Aplicación

La aplicación se puede ejecutar en dos modos: **Producción** (recomendado para uso general) y **Desarrollo** (si estás modificando el frontend).

#### Modo Producción (Recomendado)

1.  **Construye el Frontend:**
    Desde el directorio `frontend/`, ejecuta el script de build.
    ```bash
    npm run build
    ```

2.  **Inicia el Servidor Unificado:**
    Vuelve al directorio raíz del proyecto e inicia el servidor Python.
    ```bash
    python3 -m src.pipeline_server
    ```

3.  Abre tu navegador en **`http://localhost:8000`**.

#### Modo Desarrollo

Este modo te permite ver los cambios del frontend en tiempo real sin necesidad de reconstruir.

1.  **Inicia el Servedor de Desarrollo del Frontend:**
    En una terminal, navega a `frontend/` y ejecuta:
    ```bash
    npm run dev
    ```
    Esto iniciará un servidor en `http://localhost:5173`.

2.  **Inicia el Servidor Backend:**
    En **otra terminal**, desde la raíz del proyecto, inicia el servidor Python con la variable de entorno `AETHERIA_ENV`.
    ```bash
    AETHERIA_ENV=development python3 -m src.pipeline_server
    ```
    El servidor backend actuará como proxy para el frontend.

3.  Abre tu navegador en **`http://localhost:8000`**.

## 🏛️ Estructura del Proyecto

```
/
├── frontend/           # 🎨 Código fuente del frontend en React y Mantine
├── src/                # 🧠 Lógica principal del backend en Python
│   ├── pipeline_server.py  # 🚀 Punto de entrada del servidor web unificado
│   ├── server_handlers.py  #  WebSocket y lógica de control
│   ├── server_state.py     # Gestión del estado global del servidor
│   ├── qca_engine.py       # 🌌 Motor de simulación QCA
│   ├── trainer.py          # 🏋️ Lógica de entrenamiento de modelos
│   └── models/             # 🧬 Arquitecturas de las "Leyes M" (U-Net, etc.)
│
├── output/
│   └── experiments/    # 📂 Todos los resultados, organizados por experimento
│       └── {exp_name}/
│           ├── checkpoints/ # 💾 Modelos entrenados (.pth)
│           ├── simulations/ # (Futuro) Estados de simulación guardados
│           └── visualizations/ # (Futuro) Videos o gráficos generados
│
├── docs/               # 📄 Documentación detallada
└── requirements.txt    # 🐍 Dependencias de Python
```

Para más detalles sobre la arquitectura, los modelos y las estrategias de entrenamiento, consulta los documentos en la carpeta `docs/`.
