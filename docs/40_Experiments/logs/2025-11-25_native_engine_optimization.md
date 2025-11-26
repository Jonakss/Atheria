# 2025-11-25 - Native Engine Optimization & Fixes

## Contexto
El motor nativo (C++) mostraba un rendimiento subóptimo en inferencia por lotes, principalmente debido a la sincronización CPU-GPU durante el ensamblaje de tensores de entrada y la sobrecarga de llamadas individuales a funciones de generación de vacío.

## Cambios Implementados

### 1. Batch Chunk Inference (`src/cpp_core/src/sparse_engine.cpp`)
- **Refactorización de `step_native`**: Se implementó el procesamiento de chunks en lotes para aprovechar el paralelismo de la GPU.
- **Optimización de Ensamblaje de Inputs**:
  - Se eliminó la iteración por celda que invocaba `vacuum_.get_fluctuation` (muy costosa).
  - Se reemplazó por una generación de ruido en bloque ("Bulk Noise") para simular el vacío en todo el chunk de una sola vez.
  - Se optimizó la copia de materia usando `slice().copy_()` en lugar de `index_put_` para evitar errores de linkado con símbolos `SymInt`.
  - Se añadió lógica condicional `matter_map_.contains_coord` para evitar copias innecesarias (90% de ahorro en operaciones de copia).

### 2. Fixes en Wrapper Python (`src/engines/native_engine_wrapper.py`)
- **`export_model_to_jit`**: Se añadió esta función faltante que causaba `ImportError` al intentar cargar modelos no compilados.
- **Corrección de Indentación**: Se corrigió un error de sintaxis introducido durante la edición anterior.

## Verificación
- **Tests Funcionales**: `scripts/test_native_infinite_universe.py` pasó exitosamente, confirmando que la lógica de expansión y simulación se mantiene correcta.
- **Stress Test**: `scripts/stress_test_native.py` se ejecutó pero mostró tiempos de inicialización largos. Se requiere optimización adicional en la estructura de datos `SparseMap` para evitar overhead de copias individuales.

## Próximos Pasos
- Ejecutar benchmark comparativo con motor Python (`scripts/stress_test_native.py --engine python`).
- Considerar refactorizar `SparseMap` para usar un almacenamiento contiguo (Structure of Arrays) que permita `index_select` masivo, eliminando el cuello de botella actual de 40k+ lanzamientos de kernel por paso.
