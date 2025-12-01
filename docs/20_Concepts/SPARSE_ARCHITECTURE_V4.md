# Arquitectura Sparse en Atheria 4

## 📊 Resumen

**Sparse (disperso) es la arquitectura base del motor nativo C++ en Atheria 4**. El motor nativo usa `SparseMap` internamente para almacenar solo las partículas activas, no todo el grid completo.

## 🏗️ Arquitectura de Almacenamiento

### Motor Nativo C++ (Inferencia - V4)

**Formato:** Sparse (disperso)
- ✅ Usa `SparseMap` (hash map en C++) para almacenar solo coordenadas con partículas activas
- ✅ Genera vacío cuántico (`HarmonicVacuum`) on-demand para coordenadas vacías
- ✅ Mucho más eficiente en memoria y rendimiento
- ✅ Arquitectura base para inferencia en V4

**Implementación:**
- `src/cpp_core/src/sparse_engine.cpp`: Motor C++ con `SparseMap`
- `src/cpp_core/include/sparse_map.h`: Estructura de datos dispersa
- `src/engines/native_engine_wrapper.py`: Wrapper que convierte sparse ↔ dense

### Motor Python (Entrenamiento)

**Formato:** Dense (denso)
- ✅ Usa grid completo (`torch.Tensor` de tamaño `[1, H, W, d_state]`)
- ✅ Necesario para el entrenamiento (backpropagation requiere grid completo)
- ✅ Menos eficiente en memoria pero necesario para entrenamiento

**Implementación:**
- `src/engines/qca_engine.py`: Motor Python con grid denso
- Usado por `QC_Trainer_v4` durante el entrenamiento

## 🔄 Conversión Automática

El `NativeEngineWrapper` realiza la conversión automática entre formatos:

```python
# Motor nativo (sparse) → Frontend (dense)
def _update_dense_state_from_sparse(self):
    """
    Convierte el estado disperso del motor nativo a formato denso (grid)
    para compatibilidad con el frontend.
    
    Solo actualiza regiones activas cuando es posible (optimización).
    """
    # El motor nativo almacena partículas dispersas
    # El frontend necesita un grid denso
    # Se obtiene el estado desde el motor nativo (genera vacío automáticamente si no hay partícula)
```

**Flujo:**
1. **Motor nativo ejecuta** (`step_native()`): Actualiza solo partículas activas en `SparseMap`
2. **Wrapper convierte** (`_update_dense_state_from_sparse()`): Genera grid denso para visualización
3. **Frontend recibe**: Grid denso completo para renderizado

## ⚙️ Configuración

**No requiere configuración manual** - Es automático:

1. **Motor nativo C++**: Siempre usa sparse (no configurable, es su arquitectura base)
2. **Motor Python**: Siempre usa dense (necesario para entrenamiento)
3. **Conversión**: Automática cuando se usa el motor nativo con el frontend

### Cuando se usa cada uno:

| Escenario | Motor | Formato | Razón |
|-----------|-------|---------|-------|
| **Inferencia** | Nativo C++ | Sparse | ⚡ Rendimiento óptimo, memoria eficiente |
| **Entrenamiento** | Python | Dense | 📚 Necesario para backpropagation |
| **Visualización** | Cualquiera | Dense | 🎨 Frontend necesita grid completo |

## 🎯 Ventajas del Sparse en V4

1. **Memoria eficiente**: Solo almacena partículas activas (~1% del espacio en simulaciones dispersas)
2. **Rendimiento**: Evita procesar celdas vacías innecesariamente
3. **Escalabilidad**: Permite simulaciones mucho más grandes
4. **Vacío cuántico on-demand**: Genera fluctuaciones solo cuando se necesitan

## 📝 Notas Técnicas

### SparseMap (C++)

- Estructura hash map optimizada para coordenadas 3D
- Almacena `torch::Tensor` directamente en C++
- Hash personalizado para `Coord3D`

### HarmonicVacuum (C++)

- Genera fluctuaciones del vacío cuántico deterministas
- Permite que el vacío sea consistente (misma coordenada = mismo ruido)
- No requiere almacenar todo el grid

### Conversión Dense ↔ Sparse

- **Sparse → Dense**: Iterar sobre coordenadas activas + generar vacío para el resto
- **Dense → Sparse**: Filtrar celdas con energía > umbral y almacenar en `SparseMap`

## 🔗 Referencias

- `src/cpp_core/src/sparse_engine.cpp`: Implementación C++ del motor sparse
- `src/cpp_core/include/sparse_map.h`: Estructura de datos SparseMap
- `src/engines/native_engine_wrapper.py`: Wrapper con conversión automática
- `src/engines/qca_engine.py`: Motor Python denso (entrenamiento)

## Enlaces Relacionados

- [[SPARSE_ENGINE_ACTIVE_NEIGHBORS]] - Cómo se procesan vecinos activos
- [[HARMONIC_VACUUM_CONCEPT]] - Vacío armónico on-demand
- [[NATIVE_ENGINE_DEVICE_CONFIG]] - Configuración de device
- [[PYTHON_TO_NATIVE_MIGRATION]] - Migración de experimentos
- [[NATIVE_PARALLELISM]] - Paralelismo en el motor

## Tags

#sparse #architecture #native-engine #memory-optimization #cpp

---

**Estado:** ✅ Implementado y activo por defecto en motor nativo C++  
**Configuración:** Automática (no requiere configuración manual)  
**Última actualización:** 2024-11-20

