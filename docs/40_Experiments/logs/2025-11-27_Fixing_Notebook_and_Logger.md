# 2025-11-27 - Fix: Notebook Bugs, ExperimentLogger Paths & Agent Safety

## Contexto
El usuario reportó múltiples errores al intentar ejecutar el notebook de entrenamiento progresivo (`Atheria_Progressive_Training.ipynb`). Los errores incluían fallos en la inicialización del modelo, funciones faltantes y problemas con la creación de directorios de logs. Además, durante el proceso de corrección, un script de prueba borró accidentalmente documentación, lo que llevó a implementar nuevas reglas de seguridad.

## Problemas y Soluciones

### 1. Channel Mismatch en Inicialización de UNet
**Problema:** `RuntimeError` debido a discrepancia de canales (esperado 8, recibido 4).
**Causa:** `QC_Trainer_v4` se inicializaba sin pasar `model_params`, por lo que `d_state` defaultaba a 2 (y el UNet espera `2 * d_state` inputs).
**Solución:** Se actualizó la llamada al constructor de `QC_Trainer_v4` en el notebook para pasar explícitamente `model_params=phase_cfg['MODEL_PARAMS']`.

### 2. Tuple Unpacking en Training Loop
**Problema:** `TypeError` al intentar desempaquetar el retorno de `train_episode`.
**Causa:** El notebook esperaba un diccionario (`epoch_result`), pero la función retornaba una tupla `(loss, metrics)`.
**Solución:** Se actualizó el bucle de entrenamiento para desempaquetar correctamente: `loss, metrics = trainer.train_episode(episode)`.

### 3. Código en Celda Markdown
**Problema:** El bucle de entrenamiento principal no se ejecutaba.
**Causa:** El bloque de código estaba accidentalmente dentro de una celda de tipo Markdown.
**Solución:** Se convirtió la celda a tipo Code programáticamente.

### 4. Función Faltante `find_latest_checkpoint`
**Problema:** `NameError: name 'find_latest_checkpoint' is not defined`.
**Causa:** La función fue eliminada accidentalmente al borrar una sección obsoleta del notebook.
**Solución:** Se restauró la definición de la función en la celda de imports.

### 5. ExperimentLogger: Directorios Inexistentes
**Problema:** `FileNotFoundError` al intentar crear el log del experimento.
**Causa:** `ExperimentLogger` no creaba automáticamente los directorios padres si el nombre del experimento contenía subdirectorios (ej: `MultiPhase/Fase1`).
**Solución:** Se modificó `src/utils/experiment_logger.py` para asegurar la creación de `os.path.dirname(self.log_file)`.

### 6. ExperimentLogger: Rutas Relativas Incorrectas
**Problema:** Se creaba una carpeta `notebooks/docs/` errónea.
**Causa:** Al ejecutar desde el notebook, el CWD es `notebooks/`, y `ExperimentLogger` usaba rutas relativas.
**Solución:** Se implementó resolución dinámica de la ruta absoluta del proyecto en `ExperimentLogger` para asegurar que siempre escriba en `<project_root>/docs/`.

### 7. Seguridad de Agentes (Agent Guidelines)
**Problema:** Un script de prueba (`repro_logger_error.py`) borró accidentalmente `docs/40_Experiments`.
**Causa:** Uso agresivo de `shutil.rmtree()` en un directorio compartido.
**Solución:**
- Se restauraron los archivos borrados vía `git checkout`.
- Se actualizó `docs/99_Templates/AGENT_GUIDELINES.md` prohibiendo explícitamente el borrado de directorios compartidos en scripts de prueba.

## Archivos Modificados
- `notebooks/Atheria_Progressive_Training.ipynb`
- `src/utils/experiment_logger.py`
- `docs/99_Templates/AGENT_GUIDELINES.md`
- `tests/integration/test_progressive_training.py` (Nuevo test de integración)

## Estado Final
El notebook es ahora funcional y robusto. El sistema de logging es resiliente a la estructura de directorios y ubicación de ejecución. Las reglas de agente previenen futuros accidentes de borrado.
