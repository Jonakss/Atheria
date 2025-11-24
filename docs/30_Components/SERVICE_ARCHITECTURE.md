# Arquitectura Basada en Servicios

## Visión General
La arquitectura de Atheria ha evolucionado de un bucle monolítico (`SimulationLoop`) a una arquitectura basada en servicios desacoplados. Este cambio permite separar la ejecución de la física (Inferencia) del procesamiento de datos y la visualización, mejorando el rendimiento y la capacidad de respuesta.

## Componentes Principales

### 1. ServiceManager
El orquestador central que gestiona el ciclo de vida de todos los servicios.
- **Responsabilidad:** Iniciar y detener servicios en el orden correcto.
- **Ubicación:** `src/pipelines/pipeline_server.py`

### 2. SimulationService
El corazón de la simulación física.
- **Responsabilidad:** Ejecutar el bucle de física (Physics Step) utilizando el motor (Nativo o Python).
- **Características:**
    - Corre en su propio hilo/tarea asíncrona.
    - Prioriza el mantenimiento de FPS estables.
    - No se bloquea por operaciones de visualización o red.
- **Salida:** Envía estados ligeros (referencias o datos crudos) a `state_queue`.
- **Ubicación:** `src/services/simulation_service.py`

### 3. DataProcessingService
El procesador de datos pesados.
- **Responsabilidad:** Consumir estados de la simulación, generar visualizaciones y comprimir datos.
- **Características:**
    - Realiza operaciones costosas (`get_dense_state`, `generate_frame`, compresión) en un ThreadPool.
    - Puede saltar frames si la cola se llena (backpressure), evitando que la simulación se ralentice.
- **Entrada:** Consume de `state_queue`.
- **Salida:** Envía payloads listos para transmitir a `broadcast_queue`.
- **Ubicación:** `src/services/data_processing_service.py`

### 4. WebSocketService
La interfaz de comunicación.
- **Responsabilidad:** Gestionar conexiones WebSocket y transmitir datos a los clientes.
- **Características:**
    - Maneja múltiples clientes simultáneamente.
    - Enruta comandos entrantes a los handlers correspondientes.
- **Entrada:** Consume de `broadcast_queue`.
- **Ubicación:** `src/services/websocket_service.py`

## Flujo de Datos

1.  **Physics Step:** `SimulationService` avanza el motor físico.
2.  **State Push:** `SimulationService` coloca una referencia del estado en `state_queue`.
3.  **Processing:** `DataProcessingService` toma el estado, extrae datos densos, aplica mapas de color y comprime.
4.  **Broadcast Push:** `DataProcessingService` coloca el payload final en `broadcast_queue`.
5.  **Transmission:** `WebSocketService` toma el payload y lo envía a todos los clientes conectados.

## Mecanismos de Sincronización

- **Colas Asíncronas:** Se usan `asyncio.Queue` con tamaño limitado (`maxsize`) para comunicar servicios.
- **Backpressure:** Si `DataProcessingService` es lento, `state_queue` se llena y `SimulationService` descarta frames (no se bloquea), manteniendo la física en tiempo real.
- **Thread Pool:** Las operaciones intensivas de CPU (física nativa, compresión) se ejecutan en `run_in_executor` para no bloquear el Event Loop principal.
