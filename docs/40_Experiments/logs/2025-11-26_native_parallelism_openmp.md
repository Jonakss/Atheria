# 🧪 Experiment: Native Engine Parallelism (OpenMP)

**Fecha:** 2025-11-26
**Estado:** Implementado / En Verificación
**Rama:** `feat/native-parallelism`

## 🎯 Objetivo

Acelerar el bucle de simulación del motor nativo C++ (`sparse_engine.cpp`) utilizando paralelismo de CPU mediante **OpenMP**. El objetivo es escalar la simulación para soportar miles de partículas activas distribuyendo la carga de trabajo entre múltiples núcleos.

## ⚙️ Implementación

Se modificó `src/cpp_core/src/sparse_engine.cpp` para paralelizar el bucle principal de `step_native()`:

1.  **`#pragma omp parallel`**: Crea un equipo de hilos.
2.  **Thread-Local Storage**: Cada hilo tiene sus propios vectores (`local_batch_coords`, etc.) para acumular resultados parciales sin bloqueos.
3.  **`#pragma omp for schedule(dynamic)`**: Distribuye las partículas activas dinámicamente entre los hilos.
4.  **`#pragma omp critical`**: Fusiona los resultados locales en el mapa global al final del paso.

## 🚀 Cómo Usarlo

La paralelización es **automática** una vez compilado el motor. No requiere configuración explícita por parte del usuario, pero se puede ajustar mediante variables de entorno.

### Compilación
```bash
python src/cli.py build
```

### Ejecución
```bash
python src/cli.py run --frontend
```

### Ajuste de Hilos (Opcional)
Por defecto, OpenMP usa todos los núcleos disponibles. Para limitar el número de hilos (útil si compite con PyTorch):

```bash
export OMP_NUM_THREADS=4
python src/cli.py run
```

## 📊 Qué Esperar

1.  **Mayor Uso de CPU:** Deberías ver múltiples núcleos de CPU activos (usando `htop` o Monitor de Actividad) durante la simulación nativa.
2.  **Mejor FPS en Grids Grandes:** La mejora de rendimiento será más notable cuando haya **muchas partículas activas** (> 1000). En simulaciones vacías o pequeñas, el overhead de crear hilos podría no aportar beneficios visibles.
3.  **Estabilidad:**
    *   **Éxito:** La simulación corre fluida y rápida.
    *   **Fallo (Deadlock):** Si la simulación se congela totalmente (FPS = 0, no responde), puede ser un conflicto de hilos con LibTorch. En este caso, intenta reducir `OMP_NUM_THREADS`.

## 📝 Resultados Preliminares

- **Test Funcional:** `test_native_parallelism.py` pasó exitosamente (inicialización y paso de simulación correctos).
- **Benchmark:** Pendiente de realizar con carga pesada.

## 🔗 Referencias
- [[../../20_Concepts/NATIVE_PARALLELISM|Concepto: Paralelismo Nativo]]
- [[../../10_core/ROADMAP_PHASE_2|Roadmap Fase 2]]
