# ⚡ Paralelismo en Motor Nativo (OpenMP)

## 📖 Concepto

El **Paralelismo en el Motor Nativo** se refiere a la capacidad del motor C++ de Atheria para procesar múltiples partículas o regiones del espacio simultáneamente utilizando múltiples hilos de CPU. Esto es crucial para escalar la simulación a millones de partículas activas.

## 🛠️ Implementación Técnica

Utilizamos **OpenMP** (Open Multi-Processing), una API estándar para programación de memoria compartida en C++.

### Estrategia de Paralelización

La función principal `step_native()` itera sobre todas las coordenadas activas para calcular su evolución. Esta iteración es "embarrassingly parallel" (vergonzosamente paralela), ya que el estado siguiente de una celda depende solo del estado actual de sus vecinos (que es inmutable durante el paso).

```cpp
#pragma omp parallel
{
    // Almacenamiento local por hilo (Thread-Local Storage)
    std::vector<Coord3D> local_batch_coords;
    // ...
    
    #pragma omp for schedule(dynamic)
    for (size_t i = 0; i < processed_coords.size(); i++) {
        // Procesamiento independiente de cada coordenada
        // ...
    }
    
    // Fusión segura (Critical Section)
    #pragma omp critical
    {
        // Unir resultados locales al mapa global del siguiente paso
    }
}
```

### Componentes Clave

1.  **Thread-Local Storage:** Cada hilo mantiene sus propios buffers (`local_batch_coords`, `local_next_matter_map`) para evitar condiciones de carrera y contención de bloqueo.
2.  **Batch Processing:** Las partículas se agrupan en lotes (batches) para aprovechar la eficiencia de inferencia de PyTorch/LibTorch.
3.  **Dynamic Scheduling:** Usamos `schedule(dynamic)` porque la carga de trabajo por partícula puede variar (algunas pueden estar vacías o requerir menos cómputo).

## ⚠️ Consideraciones de Seguridad (Deadlocks)

Existe un riesgo conocido de **deadlocks** al combinar OpenMP con LibTorch (PyTorch C++ API), ya que ambos intentan gestionar el pool de hilos.

- **Síntoma:** La simulación se congela completamente.
- **Solución:** Ajustar `OMP_NUM_THREADS` y `torch::set_num_threads` para evitar conflictos. Generalmente, queremos que OpenMP maneje el paralelismo de alto nivel (partículas) y PyTorch use un solo hilo por operación de inferencia (ya que estamos paralelizando *fuera* de PyTorch).

## 📊 Impacto en Rendimiento

- **Esperado:** Mejora lineal con el número de núcleos físicos (hasta cierto punto de saturación de memoria).
- **Cuello de Botella:** La fusión final de resultados (`#pragma omp critical`) y la transferencia de memoria.

## 🔗 Referencias

- [[SPARSE_ENGINE_ACTIVE_NEIGHBORS]] - Cómo funcionan los vecinos activos
- [[SPARSE_ARCHITECTURE_V4]] - Arquitectura sparse general
- [[NATIVE_ENGINE_DEVICE_CONFIG]] - Configuración de device
- [[CUDA_CONFIGURATION]] - Configuración de CUDA

## Tags

#native-engine #parallelism #openmp #performance #cpp
