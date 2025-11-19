Arquitectura Técnica: Atheria 4

1. Estructura de Directorios

src/
├── engines/           # Motores de Simulación
│   ├── harmonic_engine.py  # Motor Disperso con Vacío QFT (Producción)
│   └── qca_engine.py       # Motor Denso (Laboratorio/Entrenamiento)
├── models/            # Cerebros (PyTorch)
│   ├── unet_unitary.py     # Arquitectura Principal
│   └── snn_unet.py         # Arquitectura Spiking
├── physics/           # Reglas y Entorno
│   ├── noise.py            # Inyector de Ruido IonQ
│   └── genesis.py          # Semillas iniciales (Big Bang)
├── analysis/          # Xenobiología
│   ├── epoch_detector.py   # Clasificador de Eras
│   └── structure_finder.py # (Pendiente) Detector de Gliders
└── pipeline_server.py # Servidor Orquestador (aiohttp)


2. Flujo de Datos (Data Flow)

A. Entrenamiento (The Lab)

Trainer -> Dense Engine -> Noise Injector -> U-Net -> Loss (Stability + Symmetry) -> Update Weights

B. Inferencia (The Universe)

Sparse Engine

Step Start: Identificar coordenadas activas.

Vacuum Gen: Calcular valor del vacío (HarmonicVacuum) para vecinos vacíos.

Local Tensor: Construir mini-grids (3x3) alrededor de la materia.

Inference: Pasar mini-grids por la U-Net (pre-entrenada).

Update: Actualizar Hash Map matter. Borrar celdas con $E < \epsilon$.

Viewport: Muestrear región de cámara -> Enviar a Frontend.

3. Stack Tecnológico

Backend: Python 3.10+, PyTorch (CUDA), Aiohttp.

Estructuras: Diccionarios nativos de Python (prototipo) -> torch.sparse (futuro).

Frontend: React, Three.js (HolographicViewer), WebSockets.

4. Reglas de Programación (Guidelines)

Tensor Agnostic: El código debe intentar funcionar tanto para 2D (B, C, H, W) como para 3D (B, C, D, H, W) siempre que sea posible.

No "Magic Numbers": Las constantes físicas ($\gamma$, noise_rate) deben venir del CurriculumManager, no estar hardcodeadas.

Visualize First: Antes de optimizar, asegura que el HolographicViewer pueda renderizar el estado.