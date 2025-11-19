# Arquitectura del Sistema Aetheria

Este documento describe la arquitectura de alto nivel de Aetheria, dividida en sus componentes principales: el Backend, el Frontend y el flujo de comunicación entre ellos.

## 1. Componente Backend (`src/`)

El backend es el cerebro de Aetheria. Está construido en Python utilizando `aiohttp` para la comunicación asíncrona y `torch` para la simulación y el entrenamiento.

-   **`pipeline_server.py`**: Es el punto de entrada principal de la aplicación (`python3 -m src.pipeline_server`). Sus responsabilidades son:
    -   Configurar y ejecutar el servidor web `aiohttp`.
    -   Definir las rutas principales: `/ws` para el WebSocket y las rutas para servir el frontend.
    -   Manejar el modo de ejecución (Producción vs. Desarrollo).

-   **`server_handlers.py`**: Contiene toda la lógica de la aplicación.
    -   **`websocket_handler`**: Gestiona la conexión WebSocket. Recibe comandos de la UI (ej. `start_training`, `load_simulation`) y envía actualizaciones de estado, métricas e imágenes de la simulación.
    -   **Lógica de Control**: Funciones como `start_simulation`, `stop_simulation`, `start_training` que inician y detienen los procesos principales.
    -   **Bucles Asíncronos**: El bucle de simulación (`simulation_loop`) y el de visualización (`visualization_loop`) se ejecutan como tareas de fondo (`asyncio.Task`) gestionadas por el servidor.

-   **`server_state.py`**: Define y gestiona el estado global de la aplicación en una clase `ServerState`. Esto evita el uso de variables globales desorganizadas y centraliza el estado de la simulación, el entrenamiento y la configuración.

-   **`qca_engine.py` (`Aetheria_Motor`)**: Es el motor de física fundamental.
    -   Contiene el estado de la grilla (`psi`).
    -   Carga el modelo de PyTorch (la "Ley M").
    -   Ejecuta un paso de simulación (`tick()`) aplicando el modelo para calcular el siguiente estado de la grilla.

-   **`trainer.py` (`QC_Trainer_v3`)**: Gestiona el ciclo de entrenamiento de un modelo.
    -   Implementa el bucle de entrenamiento por episodios.
    -   Calcula la función de recompensa.
    -   Realiza la retropropagación (BPTT).
    -   Guarda y carga checkpoints en la estructura de directorios `output/experiments/`.

-   **`models/`**: Contiene las diferentes arquitecturas de red neuronal que pueden actuar como "Ley M". Cada archivo define una clase que hereda de `torch.nn.Module`.

## 2. Componente Frontend (`frontend/`)

El frontend es la interfaz de usuario interactiva para controlar y visualizar el laboratorio. Está construido con React, Vite y la librería de componentes Mantine.

-   **`main.tsx`**: El punto de entrada de la aplicación React. Renderiza el componente principal `App`.

-   **`App.tsx`**: El componente raíz que ensambla toda la interfaz.
    -   Gestiona el estado principal de la UI (ej. experimento seleccionado, checkpoint seleccionado).
    -   Utiliza el hook `useWebSocket` para la comunicación con el backend.
    -   Renderiza los componentes principales de la layout (`MainHeader`, `LabSider`, `MetricsDashboard`).

-   **`hooks/useWebSocket.ts`**: Un hook personalizado que encapsula toda la lógica de comunicación WebSocket.
    -   Establece y mantiene la conexión con el servidor.
    -   Proporciona una función `sendCommand` para enviar mensajes al backend.
    -   Recibe mensajes del backend y actualiza el estado de React (métricas, imágenes, logs, etc.), haciendo que la UI sea reactiva.

-   **`components/`**: Contiene los componentes reutilizables de la UI.
    -   **`LabSider.tsx`**: La barra lateral de control desde donde se gestionan los experimentos, se carga la simulación y se inicia el entrenamiento.
    -   **`MetricsDashboard.tsx`**: El panel que muestra los gráficos de entrenamiento, los logs y las métricas en tiempo real.
    -   **`PanZoomCanvas.tsx`**: El componente de visualización principal que renderiza la imagen de la simulación y permite la interacción (pan y zoom).

## 3. Flujo de Comunicación (WebSocket)

La comunicación entre el frontend y el backend se realiza exclusivamente a través de mensajes JSON sobre una conexión WebSocket.

-   **Frontend a Backend (Comandos)**: La UI envía objetos JSON con una estructura `{ "scope": "...", "command": "...", "args": {...} }`.
    -   `scope`: Puede ser `'sim'` (control del simulador) o `'lab'` (control del entrenamiento/experimentos).
    -   `command`: La acción a realizar (ej. `'start'`, `'stop_training'`, `'refresh_checkpoints'`).

-   **Backend a Frontend (Actualizaciones)**: El servidor envía objetos JSON con una estructura `{ "type": "...", "payload": {...} }`.
    -   `type`: El tipo de actualización (ej. `'image_update'`, `'metrics_update'`, `'training_log'`).
    -   `payload`: Los datos correspondientes a la actualización.

Este flujo permite que la UI sea un cliente "tonto" que simplemente envía comandos y reacciona a las actualizaciones de estado que el servidor le proporciona.
