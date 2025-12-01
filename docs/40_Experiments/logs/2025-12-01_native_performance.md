# 2025-12-01 - Performance: Native Engine Patch Size Optimization

**Tipo:** `perf`

## Descripción
Se detectó y corrigió un cuello de botella crítico en `sparse_engine.cpp`. La inferencia local estaba utilizando el tamaño completo del grid (128x128) para construir parches, en lugar de una ventana local de 3x3.

## Cambios
- Reducción de `patch_size` de `grid_size` a 3.
- Corrección de índices de centro para el tensor de salida.

## Impacto
- Reducción drástica de operaciones redundantes en `step_native`.
- Habilita el benchmarking real del paralelismo.
