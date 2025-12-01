# Tests del Motor Nativo C++

**Componente:** `src/cpp_core/`
**Ubicación Tests:** `tests/`
**Fecha:** 2025-12-01

---

## 🎯 Objetivo

Documentar los procedimientos de prueba para verificar la corrección, estabilidad y rendimiento del Motor Nativo C++ (`atheria_core`).

## 🧪 Tests Disponibles

### 1. Test de Integración Octree (`test_octree_integration.py`)

**Propósito:**
Verificar que la integración del Octree en el motor nativo funciona correctamente para consultas espaciales y que `step_native` utiliza el ordenamiento Morton sin errores.

**Ejecución:**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/ath_venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib
python3 tests/test_octree_integration.py
```

**Verificaciones:**
- `query_radius`: Confirma que la consulta de radio devuelve las coordenadas correctas dentro de un bounding box.
- `step_native`: Ejecuta pasos de simulación para asegurar que el ordenamiento espacial no rompe la lógica.

### 2. Test de Memory Pools (`test_memory_pool.py`)

**Propósito:**
Verificar la estabilidad y corrección del sistema de `TensorPool`. Asegura que la reutilización de tensores no introduce corrupción de datos ni fugas de memoria.

**Ejecución:**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/ath_venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib
python3 tests/test_memory_pool.py
```

**Verificaciones:**
- **Estabilidad:** Ejecuta 50 pasos de simulación para detectar segfaults o errores de memoria.
- **Carga de Modelo:** Verifica que el modelo TorchScript se carga y ejecuta correctamente.
- **Concurrencia:** Valida que no existen deadlocks en la ejecución paralela (OpenMP), especialmente en la generación de ruido (`HarmonicVacuum`).

## 🛠️ Troubleshooting Común

### Error: `ModuleNotFoundError: No module named 'atheria_core'`
- **Causa:** El módulo C++ no está instalado o no está en el `PYTHONPATH`.
- **Solución:**
  ```bash
  pip install .
  ```

### Error: `ImportError: ... libnvJitLink.so.12 ...`
- **Causa:** LibTorch no encuentra las librerías de CUDA/NVIDIA.
- **Solución:** Exportar `LD_LIBRARY_PATH` apuntando al entorno virtual.
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/ath_venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib
  ```

### Hang / Congelamiento en `step_native`
- **Causa:** Posible deadlock en OpenMP.
- **Diagnóstico:** Ejecutar con un solo hilo para descartar problemas de concurrencia.
  ```bash
  export OMP_NUM_THREADS=1
  python3 tests/test_memory_pool.py
  ```
- **Nota:** Si funciona con 1 hilo pero falla con múltiples, revisar `HarmonicVacuum` y generadores de números aleatorios.

## 🔄 Ciclo de Desarrollo

1.  **Modificar C++:** Editar archivos en `src/cpp_core/`.
2.  **Recompilar:** `pip install .`
3.  **Ejecutar Tests:** Correr los scripts de prueba relevantes.
4.  **Verificar:** Asegurar que todos los tests pasan (salida ✅).
