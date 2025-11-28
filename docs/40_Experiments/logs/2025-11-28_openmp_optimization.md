# 2025-11-28: Optimización de Motor Nativo con OpenMP

## Objetivo
Activar y verificar el paralelismo OpenMP en el motor nativo de C++ (`atheria_core`) para mejorar el rendimiento de la simulación en CPU.

## Cambios Realizados

### 1. Verificación de Configuración
- Se confirmó que `CMakeLists.txt` ya incluía la configuración necesaria para buscar y enlazar OpenMP (`find_package(OpenMP REQUIRED)`).
- Se verificó que `src/cpp_core/src/sparse_engine.cpp` utiliza pragmas de OpenMP (`#pragma omp parallel`, `#pragma omp for`).

### 2. Exposición de Control de Hilos
- Se modificó `src/cpp_core/src/bindings.cpp` para exponer las funciones de OpenMP a Python:
    - `set_num_threads(int)`: Wrapper para `omp_set_num_threads`.
    - `get_max_threads()`: Wrapper para `omp_get_max_threads`.
- Se reconstruyó la extensión usando `python setup.py build_ext --inplace --force`.

### 3. Wrapper de Python
- Se actualizó `src/engines/native_engine_wrapper.py` para incluir funciones helper que llaman a las bindings nativas de forma segura:
    - `set_num_threads(num_threads)`
    - `get_max_threads()`

### 4. Verificación
- Se verificó que el módulo compilado (`.so`) está enlazado dinámicamente con `libgomp` usando `ldd`.
- Se ejecutó un script de prueba que confirmó la capacidad de cambiar el número de hilos desde Python.

## Resultados
- El motor nativo ahora soporta paralelismo OpenMP configurable desde Python.
- Esto permite ajustar el uso de CPU según la carga y el entorno (ej. limitar hilos en entornos compartidos).

## Próximos Pasos
- Integrar el control de hilos en la UI del Dashboard (opcional).
- Realizar benchmarks comparativos con diferentes números de hilos.
