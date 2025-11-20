# Motor Nativo C++ - Atheria Core

**Componente:** `src/cpp_core/`  
**Fecha:** 2024-12-XX  
**Objetivo:** Motor de simulación de alto rendimiento en C++ para escalar de miles a millones de partículas.

---

## 🎯 Propósito

El motor nativo C++ ejecuta completamente en C++ utilizando LibTorch, eliminando el overhead del intérprete Python y permitiendo:

- **Escalabilidad:** Manejar millones de partículas activas
- **Rendimiento:** Ejecución directa en GPU sin transferencias CPU↔GPU innecesarias
- **Memoria:** Almacenamiento disperso eficiente con `SparseMap`
- **Batch Processing:** Procesamiento por batches optimizado

---

## ⚙️ Arquitectura

### Enfoque Disperso vs Denso

**Python (Denso):**
- Estado completo en un tensor `(1, grid_size, grid_size, d_state)`
- Todos los puntos del grid están en memoria
- Útil para visualización y grids pequeños

**C++ (Disperso):**
- Solo almacena partículas activas en `SparseMap`
- Genera vacío proceduralmente con `HarmonicVacuum`
- Útil para simulaciones grandes con pocas partículas

**Conversión:**
- `NativeEngineWrapper` convierte disperso → denso para visualización
- Frontend siempre recibe grid denso (compatibilidad)

---

## 🔧 Componentes Principales

### 1. SparseMap

**Ubicación:** `src/cpp_core/include/sparse_map.h`

**Funcionalidad:**
- Almacenamiento disperso usando `std::unordered_map<Coord3D, torch::Tensor>`
- Operaciones con coordenadas 3D y tensores PyTorch
- Compatibilidad hacia atrás con valores numéricos

**Uso:**
```cpp
SparseMap map;
Coord3D coord(10, 20, 0);
torch::Tensor state = ...;
map.insert_tensor(coord, state);
torch::Tensor retrieved = map.get_tensor(coord);
```

### 2. HarmonicVacuum

**Ubicación:** `src/cpp_core/include/sparse_engine.h`

**Funcionalidad:**
- Genera fluctuaciones cuánticas deterministas
- Estados complejos usando `torch::complex(cos(noise), sin(noise))`
- Semillas deterministas basadas en coordenadas y tiempo

**Implementación:**
```cpp
torch::Tensor noise = torch::randn({d_state}) * 0.1;
torch::Tensor real = torch::cos(noise);
torch::Tensor imag = torch::sin(noise);
return torch::complex(real, imag);
```

### 3. Engine

**Ubicación:** `src/cpp_core/include/sparse_engine.h`

**Funcionalidad:**
- Motor principal de simulación
- Carga modelos TorchScript (`.pt`)
- Ejecuta `step_native()` completamente en C++
- Batch processing optimizado

**Flujo de `step_native()`:**
1. Identificar coordenadas activas
2. Agrupar en batches (tamaño 32)
3. Construir patches 3x3 para cada partícula
4. Ejecutar inferencia neuronal (LibTorch)
5. Procesar salida (delta_real, delta_imag → complejo)
6. Aplicar evolución: `new_state = current_state + delta`
7. Normalizar para conservación de probabilidad
8. Filtrar estados con energía baja (< 0.01)
9. Actualizar mapa disperso

---

## 📥 Inputs / 📤 Outputs

### Engine::step_native()

**Input:**
- `matter_map_`: Mapa disperso de partículas activas
- `model_`: Modelo TorchScript cargado
- `active_region_`: Coordenadas activas para procesar

**Output:**
- `next_matter_map_`: Nuevo mapa disperso actualizado
- `next_active_region_`: Nuevas coordenadas activas
- `int64_t`: Número de partículas después del paso

**Formato de Tensores:**
- **Estado:** `torch::Tensor` complejo, shape `[d_state]`
- **Batch Input:** `[batch, 2*d_state, 3, 3]` (patch 3x3, real+imag concatenado)
- **Batch Output:** `[batch, 2*d_state, 3, 3]` (delta real+imag)
- **Delta Complejo:** `torch::complex(delta_real, delta_imag)` shape `[d_state]`

---

## 🔗 Dependencias

**Importa de:**
- LibTorch (`torch/torch.h`, `torch/script.h`)
- PyBind11 (`pybind11/pybind11.h`)
- STL (`unordered_map`, `vector`, etc.)

**Usado por:**
- `NativeEngineWrapper` (Python) - Interface compatible con `Aetheria_Motor`
- `pipeline_server.py` - Puede usar motor nativo opcionalmente

---

## 📝 Notas de Implementación

### 1. Conversión Disperso ↔ Denso

**Disperso → Denso:**
- `NativeEngineWrapper._update_dense_state_from_sparse()`
- Itera sobre todo el grid y obtiene estado desde motor nativo
- Motor nativo genera vacío automáticamente si no hay partícula

**Denso → Disperso:**
- No implementado (motor nativo inicializa disperso)
- Puede agregarse para importar estados densos existentes

### 2. Batch Processing

**Optimización:**
- Procesa en batches de 32 partículas
- Reduce overhead de llamadas a LibTorch
- Puede ajustarse según memoria disponible

**Construcción de Patch:**
- Para cada partícula, construye patch 3x3 de vecinos
- Obtiene estados (materia o vacío) para cada vecino
- Convierte estados complejos a `[real, imag]` concatenado

### 3. Normalización

**Conservación de Probabilidad:**
- Normaliza después de aplicar delta: `norm = sum(|state|²)`
- Divide por `sqrt(norm)` si `norm > 1e-6`
- Asegura conservación de probabilidad cuántica

### 4. Filtrado de Energía

**Umbral de Existencia:**
- Solo almacena estados con `energy > 0.01`
- Filtra fluctuaciones del vacío muy pequeñas
- Reduce crecimiento exponencial del mapa disperso

---

## 🚀 Uso desde Python

```python
import atheria_core
from src.engines.native_engine_wrapper import NativeEngineWrapper

# Crear wrapper (interface compatible)
wrapper = NativeEngineWrapper(grid_size=128, d_state=8, device="cpu")

# Cargar modelo TorchScript
wrapper.load_model("path/to/model.pt")

# Agregar partículas iniciales
wrapper.add_initial_particles(num_particles=10)

# Evolucionar estado
wrapper.evolve_internal_state()

# Acceder al estado denso (para visualización)
psi = wrapper.state.psi  # Tensor complejo [1, 128, 128, 8]
```

---

## 🐛 Issues Conocidos

### 1. Runtime CUDA Error

**Problema:**
```
undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12
```

**Causa:**
- Conflictos de versiones CUDA / LibTorch
- Configuración de `LD_LIBRARY_PATH`

**Solución:**
- Usar CPU temporalmente: `device="cpu"`
- O resolver dependencias CUDA correctamente
- No crítico para funcionalidad básica

### 2. Conversión Disperso ↔ Denso

**Overhead:**
- Conversión completa puede ser costosa para grids grandes
- Podría optimizarse iterando solo sobre coordenadas activas

**Mejora Futura:**
- Frontend podría recibir formato disperso directamente
- Reducir tamaño de transferencia WebSocket

---

## 📊 Métricas de Rendimiento

**Objetivo:**
- Python: ~1000 partículas máximo en tiempo real
- C++: ~100,000+ partículas en tiempo real (objetivo)

**Benchmark Pendiente:**
- Comparar `Aetheria_Motor` (Python) vs `Engine` (C++)
- Medir tiempo de `step()` para diferentes tamaños
- Medir uso de memoria

---

## 🔗 Referencias

- [[PHASE_2_SETUP_LOG]] - Log de setup inicial
- [[AI_DEV_LOG#2024-12-XX - Fase 2 Iniciada]] - Documentación de decisiones
- `src/engines/native_engine_wrapper.py` - Wrapper Python
- `src/cpp_core/src/sparse_engine.cpp` - Implementación C++

---

**Estado:** ✅ **Implementación Básica Completada**  
**Próximos Pasos:** Tests con modelos reales, benchmarking, optimizaciones

