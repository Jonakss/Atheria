Arquitectura Técnica: Atheria 4

1. Estructura de Directorios

src/
├── cpp_core/          # Motor Nativo C++ (Producción)
│   ├── include/       # Headers (.h)
│   └── src/           # Implementación (.cpp, .cu)
├── engines/           # Motores de Simulación
│   ├── backends/           # Backends de Cómputo (Hardware Abstraction)
│   │   ├── compute_backend.py  # Interfaz Base + IonQBackend
│   │   └── ...
│   ├── native_engine_wrapper.py # Wrapper Híbrido (Native -> Python Fallback)
│   ├── harmonic_engine.py  # Motor Disperso Python (Legacy/Fallback)
│   └── qca_engine.py       # Motor Denso (Laboratorio/Entrenamiento)
├── models/            # Cerebros (PyTorch)
│   ├── unet_unitary.py     # Arquitectura Principal
│   ├── snn_unet.py         # Arquitectura Spiking
│   └── layers/             # Capas Personalizadas
│       └── holographic.py  # (Prototipo) Capa Convolucional Cuántica
├── physics/           # Reglas y Entorno
│   ├── noise.py            # Inyector de Ruido IonQ
│   └── genesis.py          # Semillas iniciales (Big Bang)
├── analysis/          # Xenobiología
│   ├── analysis.py         # t-SNE Atlas & Mapa Químico
│   └── epoch_detector.py   # (Planificado) Clasificador de Eras
└── run_server.py      # Entry Point del Servidor (usado por CLI)


2. Flujo de Datos (Data Flow)

A. Entrenamiento (The Lab)

Trainer -> Dense Engine (QCA) -> Noise Injector -> U-Net -> Loss (Stability + Symmetry) -> Update Weights

C. Computación Cuántica (IonQ Integration)

Engine -> ComputeBackend (IonQBackend) -> Qiskit/API -> IonQ QPU/Simulator -> Results -> Engine

B. Inferencia (The Universe) - Arquitectura Híbrida

Native Engine (C++) / Sparse Engine (Python)

*Nota: El sistema prioriza el Motor Nativo (C++/CUDA). Si no está disponible, hace fallback transparente al Motor Python.*

Step Start: Identificar coordenadas activas (Octree/Hash Map).

Vacuum Gen: Calcular valor del vacío para vecinos vacíos.

Local Tensor: Construir mini-grids (3x3) alrededor de la materia.

Inference: Pasar mini-grids por la U-Net (pre-entrenada).

Update: Actualizar estado disperso. Borrar celdas con $E < \epsilon$.

Viewport: Muestrear región de cámara (Lazy Conversion) -> Enviar a Frontend.

D. Cache Buffering (Streaming) - Desacoplamiento

Simulation Service -> Data Processing -> Dragonfly Cache (List) -> WebSocket Service (Stream Loop) -> Frontend

*Nota: Permite que la simulación corra a máxima velocidad (e.g. 100 FPS) mientras el frontend visualiza a una tasa constante (e.g. 30 FPS), evitando saturación.*

3. Stack Tecnológico

Backend: Python 3.10+, PyTorch (CUDA), Aiohttp.
Core Engine: C++17 (CUDA opcional) para simulación de alto rendimiento.

Estructuras:
- C++: Octree / Hash Map optimizado.
- Python: `native_engine_wrapper.py` gestiona la conversión a Tensors densos para visualización.

Frontend: React, Three.js (HolographicViewer - En Progreso), WebSockets.

4. Reglas de Programación (Guidelines)

Tensor Agnostic: El código debe intentar funcionar tanto para 2D (B, C, H, W) como para 3D (B, C, D, H, W) siempre que sea posible.

No "Magic Numbers": Las constantes físicas ($\gamma$, noise_rate) deben venir del CurriculumManager, no estar hardcodeadas.

Visualize First: Antes de optimizar, asegura que el HolographicViewer pueda renderizar el estado.