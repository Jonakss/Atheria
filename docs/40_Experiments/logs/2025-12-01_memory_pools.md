# 2025-12-01 - Memory Pools & Concurrency Fixes

**Status**: Completed
**Component**: Native Engine (C++)

Implementación de sistema de reciclaje de tensores y corrección de errores de concurrencia crítica.

## Cambios
1.  **TensorPool**: Implementada clase `TensorPool` para reutilizar memoria de tensores `torch::Tensor` entre pasos de simulación.
    - Reduce overhead de `malloc/free` en cada frame.
    - Integrado en `step_native` con lógica `acquire/release`.
2.  **HarmonicVacuum Fix**:
    - **Problema**: `step_native` se congelaba aleatoriamente en ejecución paralela.
    - **Diagnóstico**: `torch::manual_seed` modifica estado global y no es thread-safe dentro de bloques OpenMP.
    - **Solución**: Reemplazado por `torch::make_generator<torch::CPUGeneratorImpl>(seed)` local para generación de ruido determinista thread-safe.

## Impacto
- Eliminación de deadlocks en simulación nativa.
- Base sólida para escalar a millones de partículas sin fragmentación excesiva.

## Ver también
- [[NATIVE_ENGINE_TESTS]] - Guía de pruebas y troubleshooting.
- [[Native_Engine_Core]] - Documentación actualizada del motor.
