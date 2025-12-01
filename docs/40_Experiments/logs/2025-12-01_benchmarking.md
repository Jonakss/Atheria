# Benchmarking: Python vs Native Engine

**Fecha**: 2025-12-01  
**Objetivo**: Comparar el rendimiento del Motor Python (PyTorch) vs Motor Nativo (C++)

## Metodología

- **Grid Size**: 128x128
- **d_state**: 64 (128 canales totales: real + imaginario)
- **Device**: CPU
- **OMP_NUM_THREADS**: 8
- **Modelo**: `dummy_native_model_128ch.pt` (MockModel con Conv2D)

### Configuración de Benchmark

- **Python Engine**: 100 steps (10 warmup)
- **Native Engine**: 200 steps (20 warmup)

## Resultados

### Python Engine (PyTorch)

| Métrica | Valor |
|---------|-------|
| **Initialization Time** | ~0.01-0.02s |
| **Average FPS** | **57-66 FPS** |
| **Backend** | PyTorch (CPU) |
| **Optimizations** | GPUOptimizer, inference_mode |

### Native Engine (C++)

| Métrica | Observación |
|---------|-------------|
| **Initialization Time** | ~0.5-0.7s (más lento debido a JIT export y transferencia de partículas) |
| **Status** | ⚠️ **Bloqueado durante warmup** |
| **Observaciones** | El motor nativo se bloquea durante `evolve_internal_state`, posiblemente debido a: |
|  | - Lock contention en `_lock` (threading.RLock) |
|  | - Conversión sparse→dense muy costosa (lazy conversion no optimizada) |
|  | - Problema con `step_native()` en C++ |

## Análisis

### Resultados Parciales

- **Python Engine**: 57-66 FPS en CPU (grid 128x128, MockModel)
- **Native Engine**: No completó el benchmark (bloqueado)

### Problemas Identificados

1. **Bloqueo del Native Engine**:
   - Se bloquea durante `evolve_internal_state()`
   - Posible deadlock en lock reentrant `_lock`
   - Conversión sparse→dense puede ser muy costosa para grids grandes

2. **Inicialización Lenta**:
   - Native Engine tarda ~0.5-0.7s vs ~0.01s del Python Engine
   - Causas:
     - JIT export del modelo
     - Transferencia de partículas iniciales (grid 128x128 → miles de partículas)
     - `_initialize_native_state_from_dense` recorre todo el grid

### Próximos Pasos

1. **Debugging del Native Engine**:
   - Investigar bloqueo en `evolve_internal_state`
   - Revisar uso de locks en `NativeEngineWrapper`
   - Optimizar transferencia de partículas iniciales

2. **Benchmarking Alternativo**:
   - Probar con grids más pequeños (64x64, 32x32)
   - Medir solo `step_native()` sin conversiones
   - Comparar con motor Python en CUDA

3. **Optimizaciones Propuestas**:
   - Eliminar conversión sparse→dense durante benchmark
   - Lazy initialization de partículas
   - Batch transfer de partículas

## Conclusiones Preliminares

- El **Python Engine** funciona correctamente y alcanza **~60 FPS en CPU** para grids de 128x128
- El **Native Engine** necesita debugging antes de poder hacer una comparación justa
- La inicialización del Native Engine es significativamente más lenta debido a la transferencia de estado inicial

## Comandos Utilizados

```bash
# Benchmark con OMP_NUM_THREADS=8
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/ath_venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib
python3 -u scripts/benchmark_comparison.py
```

## Referencias

- [[benchmark_comparison.py]]: Script de benchmarking
- [[Native_Engine_Core]]: Documentación del motor nativo
- [[NATIVE_ENGINE_TESTS]]: Tests del motor nativo
