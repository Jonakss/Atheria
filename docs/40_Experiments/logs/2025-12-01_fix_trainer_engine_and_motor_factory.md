# 2025-12-01 - Fix: Trainer Engine Type Support & Motor Factory Signature

## Contexto
El usuario reportó dos errores críticos al intentar entrenar y cargar experimentos:
1. `trainer.py` rechazaba `POLAR` como un tipo de motor válido (`invalid choice`).
2. `inference_handlers.py` fallaba con `TypeError: get_motor() got multiple values for argument 'model'` al cargar experimentos.

## Cambios Implementados

### 1. Soporte para Motor Polar en Entrenamiento
- **Archivo:** `src/trainer.py`
- **Cambio:** Se agregó `"POLAR"` a la lista de `choices` en `argparse` para el argumento `--engine_type`.
- **Cambio:** Se corrigió el warning de deprecación de PyTorch cambiando `PYTORCH_CUDA_ALLOC_CONF` a `PYTORCH_ALLOC_CONF`.

### 2. Propagación de Configuración de Motor
- **Archivo:** `src/trainers/qc_trainer_v4.py`
- **Cambio:** Se actualizó `__init__` para aceptar el argumento `engine_type` y usarlo al crear la configuración para `motor_factory`. Esto asegura que la elección del usuario (ej: POLAR) tenga precedencia sobre la configuración global.
- **Archivo:** `src/pipelines/pipeline_train.py`
- **Cambio:** Se actualizó la instanciación de `QC_Trainer_v4` para pasar el `engine_type` desde la configuración del experimento.

### 3. Corrección de Firma en Motor Factory
- **Archivo:** `src/motor_factory.py`
- **Problema:** La firma era `def get_motor(config, model: nn.Module, device):`, pero la mayoría de los callers (incluyendo `inference_handlers.py`) esperaban `get_motor(config, device, model=...)`.
- **Solución:** Se cambió la firma a `def get_motor(config, device, model: nn.Module = None):` para coincidir con el patrón de uso predominante.
- **Archivo:** `src/server/server_handlers.py`
- **Cambio:** Se actualizó la llamada a `get_motor` para coincidir con la nueva firma.

## Verificación
- Se creó un script de verificación `verify_motor.py` que confirmó que la nueva firma de `get_motor` funciona correctamente con ambos estilos de llamada (posicional y keyword).
- Se verificó que `python -m src.trainer --help` muestra `POLAR` como una opción válida.

## Impacto
- Ahora es posible entrenar modelos usando el motor `POLAR`.
- La carga de experimentos para inferencia ya no fallará debido al error de firma en `get_motor`.
- Se eliminaron warnings molestos de PyTorch al inicio del entrenamiento.
