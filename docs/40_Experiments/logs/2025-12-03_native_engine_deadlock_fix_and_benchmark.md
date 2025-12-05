# 2025-12-03: Native Engine Deadlock Fix & Benchmark

**Fecha:** 2025-12-03
**Tipo:** Critical Fix

## Deadlock Fix

**Causa:** `torch::set_num_threads(1)` llamado dentro de `#pragma omp parallel`.

**Resolución:** Removida la llamada problemática.

## Benchmark Results (32x32)

- **Python:** ~10.8 FPS
- **Native (C++):** < 0.2 FPS (CPU)
- **Análisis:** Overhead en batch construction y map access

## Estado

✅ Verificado

## Referencias

- [[BENCHMARK_PYTHON_VS_NATIVE]]
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES]]
