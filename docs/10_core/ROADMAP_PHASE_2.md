# ⚡ Roadmap Fase 2: Motor Nativo (C++ Core)

**Objetivo:** Escalar la simulación de miles a millones de partículas activas eliminando el overhead del intérprete de Python.

**Estado General:** 🟡 **~70% Completado** - Motor funcional, optimizaciones pendientes (Actualizado: 2025-11-26)

---

## 1. Estrategia de Implementación

Utilizaremos un enfoque Híbrido Embebido usando PyBind11.

Python: Orquestación, Servidor Web, Entrenamiento (PyTorch), Visualización.

C++: Estructuras de datos espaciales (Sparse Octree), Bucle principal de física, Gestión de memoria.

## 2. Componentes del Núcleo C++ (src/cpp_core)

### A. SparseMap (El Universo) ✅
**Estado:** Completado

- Reemplazo del diccionario de Python
- Estructura: `std::unordered_map<Coord3D, QuantumState>`
- **Implementado:** `src/cpp_core/src/sparse_map.h`
- Soporte para valores numéricos y tensores PyTorch
- Custom Hashing implementado

**Pendiente:** Memory Pools para evitar fragmentación de RAM

---

### B. OctreeIndex (El Acelerador) ⏳
**Estado:** Pendiente (Mencionado pero no implementado)

- Índice espacial para búsquedas rápidas
- Permite consultas como `get_particles_in_radius(r)` en tiempo $O(\log N)$ en lugar de $O(N)$
- Vital para calcular gravedad y colisiones entre chunks
- **Prioridad:** Media (optimización adicional)

---

### C. Binding (La Interfaz) ✅
**Estado:** Completado

Código de PyBind11 que expone las clases C++ a Python.

// Ejemplo conceptual
PYBIND11_MODULE(atheria_native, m) {
    pybind11::class_<Universe>(m, "Universe")
        .def("step", &Universe::step)
        .def("get_state", &Universe::get_state);
}


## 3. Interoperabilidad con PyTorch ✅
**Estado:** Completado

Para que la "Ley M" (entrenada en Python) corra en el motor C++ sin salir de la GPU:

- ✅ **Exportar Modelo:** Usar `torch.jit.trace` para guardar el modelo como `model.pt`
- ✅ **Cargar en C++:** Usar LibTorch (API C++ de PyTorch) dentro del motor nativo
- ✅ **Implementado:** Carga de modelos TorchScript en `Engine` C++
- **Ventaja:** Los tensores nunca viajan a la CPU. C++ le dice a la GPU "ejecuta esto" y recibe el resultado en VRAM

---

4. Pasos de Migración

✅ Setup del Entorno: Configurar CMake y setup.py para compilar extensiones. - **COMPLETADO (2024-12)**

✅ Hello World: Crear una función add(a, b) en C++ y llamarla desde Python. - **COMPLETADO (2024-12)**
   - Función `add()` implementada y probada
   - Estructura `Coord3D` implementada
   - Clase `SparseMap` básica implementada y funcionando

🔄 Migración de Datos: Mover la estructura de datos self.matter de Python a C++.
   - `SparseMap` con soporte para valores numéricos: ✅
   - `SparseMap` con soporte para tensores PyTorch: ✅ (implementado, pendiente pruebas completas)

🔄 Migración de Lógica: Mover la función step() a C++.
   - `Engine` clase implementada con `step_native()`: ✅ (estructura lista, pendiente pruebas)

🔄 Integración de LibTorch: Conectar la U-Net.
   - Soporte para LibTorch detectado: ✅
   - Carga de modelos TorchScript: ✅ (implementado en Engine)
   - Pendiente: Pruebas con modelo real

⏳ Optimizaciones de Rendimiento (Alta Prioridad):
   
   **A. Paralelismo (OpenMP/std::thread)**
   - Paralelizar iteración sobre partículas activas en `step_native()`
   - Usar OpenMP para loops paralelos (`#pragma omp parallel for`)
   - Thread pool para operaciones independientes
   - Paralelizar conversión disperso→denso cuando sea necesario
   - **Impacto**: 2-4x mejora en simulación multi-core
   
   **B. Optimizaciones SIMD (Vectorización)**
   - Usar instrucciones SIMD (SSE/AVX) para operaciones vectoriales
   - Optimizar cálculos de física con vectorización automática
   - Mejorar rendimiento de operaciones matemáticas (suma, producto, etc.)
   - Vectorizar operaciones sobre tensores LibTorch
   - **Impacto**: 2-3x mejora en operaciones matemáticas
   
   **C. Visualización en C++ (Motor Nativo)**
   - Implementar `compute_visualization()` en Engine C++
   - Cálculos básicos (density, phase, energy) directamente en GPU
   - Reducir overhead Python en pipeline de visualización
   - Envío directo desde GPU cuando sea posible (zero-copy)
   - **Impacto**: Reducción de ~2-5ms → ~0.5-1ms por frame
   
   **D. Envío de Datos Optimizado**
   - Evaluar envío directo desde GPU (evitar CPU)
   - Considerar shaders en frontend para procesamiento
   - Optimizar serialización para datos raw
   - **Impacto**: Reducción adicional de ~1-2ms por frame
   
   **Impacto Total Esperado**: 10-50x mejora en rendimiento de simulación
   
   **Referencias**: 
   - [[40_Experiments/ARCHITECTURE_EVALUATION_GO_VS_PYTHON|Evaluación Go vs Python]] - Análisis de optimizaciones
   - [[40_Experiments/VISUALIZATION_OPTIMIZATION_ANALYSIS|Análisis de Optimización de Visualización]] - Opciones de optimización

**Nota para Agentes:** Al implementar optimizaciones, prioriza la seguridad de memoria (Smart Pointers) y el paralelismo (OpenMP/std::thread) para el bucle de física.

---

## 5. Resumen Ejecutivo

### ✅ Completado (~70%)
- Setup del entorno (CMake, setup.py)
- Hello World y funciones básicas
- SparseMap con soporte para tensores PyTorch
- Engine con `step_native()` implementado
- HarmonicVacuum (generador de vacío cuántico)
- Integración LibTorch y carga de modelos TorchScript
- PyBind11 bindings (`atheria_core` módulo)

### ⏳ Pendiente - Alta Prioridad
- **Paralelismo (OpenMP/std::thread)** - Impacto 2-4x
- **Optimizaciones SIMD** - Impacto 2-3x
- **Visualización en C++** - Reducción ~2-5ms → ~0.5-1ms
- **Envío de Datos Optimizado** - Reducción adicional ~1-2ms

### ⏳ Pendiente - Media Prioridad
- OctreeIndex (índice espacial)
- Memory Pools (optimización de memoria)
- Benchmark completo Python vs C++

**Impacto Total Esperado:** 10-50x mejora en rendimiento de simulación

---

## Referencias

- [[PHASE_STATUS_REPORT]] - Estado de todas las fases
- [[Native_Engine_Core]] - Documentación del motor nativo
- [[ARCHITECTURE_EVALUATION_GO_VS_PYTHON]] - Análisis de optimizaciones
- [[VISUALIZATION_OPTIMIZATION_ANALYSIS]] - Opciones de optimización
- [[ROADMAP_PHASE_1]] - Fase anterior
- [[ROADMAP_PHASE_3]] - Siguiente fase

---

**Última actualización:** 2025-11-26  
**Próximos pasos:** Implementar optimizaciones de paralelismo y SIMD para alcanzar objetivo de 10-50x mejora