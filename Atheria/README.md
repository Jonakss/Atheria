# Aetheria Project Knowledge Base
## Overview
- **Launchers**:
  - `train.py`: Pipeline de entrenamiento (síncrono).
  - `main.py`: Servidor de simulación (asíncrono).
  - `viewer.html`: UI manual que conecta al servidor.
  - `requirements.txt`: Dependencias del proyecto.
- **Paquete `src/` (El Cerebro):
  - `config.py`: Parámetros globales (ej. `HIDDEN_CHANNELS`, `LR_RATE_M`) y flags de ejecución (ej. `RUN_TRAINING`).
  - `qca_engine.py`: Contiene `Aetheria_Motor` (ejecuta `evolve_step`) y `QCA_State` (maneja `x_real`, `x_imag`).
  - **Leyes M (Modelos de física)**:
    - `qca_operator_mlp.py`: MLP 1x1 (míope).
    - `qca_operator_unet.py`: U-Net (con visión regional).
  - **Pipelines**:
    - `pipeline_train.py`: Lógica de entrenamiento usando `ActiveModel`.
    - `pipeline_server.py`: Lógica del servidor de simulación usando `ActiveModel`.
  - **Entrenador**:
    - `trainer.py`: Clase `QC_Trainer_v3` con `torch.amp` (precisión mixta), BPTT y agnosticismo al modelo (usa `hasattr` para verificar `M_bias_real`).
  - **Servidor (Modo "A Mano")**:
    - Usa `aiohttp` para servir `viewer.html` (en `/`) y `websockets` para la API de comandos JSON (ej. `{'command': 'pause'}` en `/ws`).

## Tareas Específicas
- **Depuración de Tracebacks:** Analizar errores como `std::bad_alloc` (RAM), `size mismatch` (arquitectura U-Net) o `NaN gradient` (LR alto).
- **Refactorización:** Modificar código, por ejemplo, separar motor del operador.
- **Optimización:** Sugerencias como `torch.compile` o arreglos de fugas de memoria.
- **Expansión de Código:** Implementar nuevas arquitecturas de Ley M (ej. `ConvLSTM`) o servidores (ej. "Laboratorio").
- **Análisis Conceptual:** Explicaciones de técnicas como BPTT, `torch.amp` o el flujo de trabajo en Aetheria.

## Estructura de Archivos
1. `README.md`: Guía principal para el conocimiento.
2. `architecture.md`: Detalles sobre la arquitectura del proyecto.
3. `code_structure.md`: Organización de archivos y componentes principales.