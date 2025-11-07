# Arquitectura del Proyecto Aetheria
## Componentes Principales
- **Lanzadores**:
  - `train.py`: Gestiona el pipeline de entrenamiento (síncrono).
  - `main.py`: Inicia el servidor de simulación (asíncrono).
  - `viewer.html`: Interfaz de usuario manual para conectarse al servidor.
  - `requirements.txt`: Especifica dependencias del proyecto.
- **Paquete `src/` (El Cerebro)**:
  - `config.py`: Contiene parámetros globales (ej. `HIDDEN_CHANNELS`, `LR_RATE_M`) y flags de ejecución.
  - `qca_engine.py`: Maneja `Aetheria_Motor` (ejecuta `evolve_step`) y `QCA_State` (almacena `x_real`, `x_imag`).
  - **Leyes M (Modelos de física)**:
    - `qca_operator_mlp.py`: Implementación del operador MLP 1x1.
    - `qca_operator_unet.py`: Arquitectura U-Net con visión regional.
- **Pipelines**:
  - `pipeline_train.py`: Lógica de entrenamiento usando `ActiveModel`.
  - `pipeline_server.py`: Gestiona la simulación en modo servidor con `ActiveModel`.
- **Entrenador**:
  - `trainer.py`: Clase `QC_Trainer_v3` con soporte para `torch.amp`, BPTT y agnosticismo al modelo.
- **Servidor**:
  - Usa `aiohttp` para servir `viewer.html` en `/` y `websockets` para recibir comandos JSON en `/ws`.