# 2025-11-27 - Fix: Import Error in Native Engine Wrapper & Git Conflicts

## 📝 Resumen
Se corrigió un error de importación relativa en `src/engines/native_engine_wrapper.py` que impedía la carga de modelos. Además, se resolvieron conflictos de fusión en `src/pipelines/pipeline_server.py`.

## 🐛 Problema Identificado
1.  **ImportError en `native_engine_wrapper.py`:**
    *   Error: `ImportError: attempted relative import beyond top-level package`
    *   Causa: El archivo intentaba importar `config` usando `from ... import config`, lo cual sube 3 niveles, pero `config.py` está solo 2 niveles arriba desde `src/engines/`.
2.  **Conflictos en `pipeline_server.py`:**
    *   Conflictos de fusión pendientes en `src/pipelines/pipeline_server.py` debido a cambios concurrentes en la rama `feat/upload-model`.

## 🛠️ Solución Implementada
1.  **Corrección de Import:**
    *   Se cambió `from ... import config` a `from .. import config` en `src/engines/native_engine_wrapper.py`.
    *   Verificado con script de prueba `verification/verify_import.py`.
2.  **Resolución de Conflictos:**
    *   Se resolvieron manualmente los conflictos en `src/pipelines/pipeline_server.py`, favoreciendo los cambios entrantes (Incoming) que incluían mejoras de estructura y seguridad, pero preservando comentarios relevantes de HEAD.
    *   Se verificó la sintaxis y se completó el merge.

## 📂 Archivos Afectados
*   `src/engines/native_engine_wrapper.py`
*   `src/pipelines/pipeline_server.py`

## ✅ Verificación
*   Script `verification/verify_import.py` confirmó que el import ahora funciona correctamente.
*   `git status` muestra el árbol limpio después del merge.
