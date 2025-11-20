# 🚀 Migración al Motor Nativo C++ - Guía Completa

**Fecha:** 2024-12-XX  
**Objetivo:** Completar la migración del motor Python al motor nativo C++ para mejorar rendimiento.

---

## 📊 Estado Actual

### ✅ Completado

1. **Motor Nativo C++ Implementado:**
   - ✅ `Engine` clase base con `step_native()`
   - ✅ `SparseMap` para almacenamiento disperso
   - ✅ `HarmonicVacuum` para generación de vacío cuántico
   - ✅ `Coord3D` para coordenadas 3D
   - ✅ Bindings PyBind11 para Python

2. **Integración con Backend:**
   - ✅ `NativeEngineWrapper` que envuelve el motor C++
   - ✅ Compatibilidad con `Aetheria_Motor` (interfaz compatible)
   - ✅ Exportación automática a TorchScript desde checkpoints
   - ✅ Carga automática de modelos JIT en motor nativo
   - ✅ Fallback a motor Python si falla el nativo

3. **Configuración:**
   - ✅ `USE_NATIVE_ENGINE = True` por defecto en `config.py`
   - ✅ Detección automática de disponibilidad
   - ✅ Manejo de errores robusto

### ⏳ Pendiente

1. **Optimizaciones:**
   - ⏳ Optimizar conversión disperso ↔ denso para visualización
   - ⏳ Batch processing más eficiente en `step_native()`
   - ⏳ Paralelización con OpenMP o CUDA

2. **Funcionalidades:**
   - ⏳ Soporte completo para modelos ConvLSTM (estados de memoria)
   - ⏳ Gestión de ROI (Region of Interest) en motor nativo
   - ⏳ Snapshots y análisis en motor nativo

3. **Testing:**
   - ⏳ Tests unitarios completos
   - ⏳ Benchmarks de rendimiento Python vs C++
   - ⏳ Tests de regresión

---

## 🎯 Pasos para Completar la Migración

### Paso 1: Verificar Compilación

```bash
# Verificar que el módulo está compilado
python3 -c "import atheria_core; print('✅ Módulo nativo disponible')"

# Si no está disponible, compilar:
python3 setup.py build_ext --inplace

# O usando pip:
pip install -e .
```

### Paso 2: Probar con Modelos Reales

```bash
# Ejecutar test con un modelo entrenado
python scripts/test_native_engine.py --experiment UNET_32ch_D5_LR2e-5

# Con más pasos para medir rendimiento
python scripts/test_native_engine.py --experiment UNET_32ch_D5_LR2e-5 --steps 100
```

**Verificaciones:**
- ✅ Modelo se exporta a TorchScript correctamente
- ✅ Motor nativo carga el modelo sin errores
- ✅ Simulación ejecuta pasos correctamente
- ✅ Estado cuántico se genera y actualiza correctamente
- ✅ Rendimiento mejorado vs Python

### Paso 3: Verificar en Producción

1. **Cargar un experimento desde el frontend:**
   - El motor nativo debería activarse automáticamente
   - Verificar logs: "✅ Motor nativo (C++) cargado exitosamente"

2. **Ejecutar simulación:**
   - Verificar que los pasos se ejecutan correctamente
   - Comparar rendimiento (FPS, tiempo por paso)

3. **Verificar visualizaciones:**
   - Asegurar que el estado se visualiza correctamente
   - Verificar que las métricas se calculan bien

### Paso 4: Optimizar Conversión Denso ↔ Disperso

**Problema Actual:**
El motor nativo usa formato disperso (solo partículas activas), pero el frontend necesita formato denso (grid completo) para visualización.

**Solución:**
- Optimizar `_update_dense_state_from_sparse()` en `NativeEngineWrapper`
- Solo actualizar regiones activas en lugar de todo el grid
- Usar batching más eficiente

### Paso 5: Benchmarks y Documentación

1. **Crear script de benchmark:**
   ```python
   # scripts/benchmark_native_vs_python.py
   # Compara rendimiento entre motor Python y C++
   ```

2. **Documentar resultados:**
   - Tiempo por paso (Python vs C++)
   - Memoria usada
   - Throughput (pasos/segundo)

---

## 🔍 Verificación de Estado

### Verificar que el Motor Nativo Está Activo

1. **En los logs del servidor:**
   ```
   ✅ Motor nativo (C++) cargado exitosamente con modelo JIT
   ⚡ Motor nativo cargado (250-400x más rápido)
   ```

2. **En el código:**
   ```python
   # Verificar en pipeline_server.py línea ~952
   if jit_path and os.path.exists(jit_path):
       motor = NativeEngineWrapper(...)
       if motor.load_model(jit_path):
           is_native = True  # ✅ Motor nativo activo
   ```

3. **En el frontend:**
   - Si el motor nativo está activo, debería haber mejor rendimiento
   - FPS más altos
   - Menos latencia

---

## 🐛 Troubleshooting

### Error: "atheria_core no está disponible"

**Solución:**
```bash
# Compilar el módulo
python setup.py build_ext --inplace

# Verificar que se generó el .so
ls -lh atheria_core*.so
```

### Error: "Modelo JIT no encontrado"

**Solución:**
- El sistema intenta exportar automáticamente
- Verificar que el checkpoint existe
- Verificar que `export_model_to_jit` funciona

### Error: "Error cargando modelo en motor nativo"

**Causas posibles:**
- Modelo TorchScript corrupto
- Incompatibilidad de versiones LibTorch
- Problemas de CUDA runtime

**Solución:**
```bash
# Probar con CPU primero
python scripts/test_native_engine.py --experiment ... --device cpu

# Verificar versiones
python -c "import torch; print(torch.__version__)"
```

### Motor Nativo Más Lento que Python

**Posibles causas:**
- Conversión disperso ↔ denso muy costosa
- Modelo no optimizado para TorchScript
- Debug mode activado

**Solución:**
- Verificar que está compilado en Release mode
- Optimizar `_update_dense_state_from_sparse()`
- Usar ROI para reducir conversión

---

## 📝 Checklist de Migración

- [ ] Verificar que el módulo C++ está compilado
- [ ] Probar con modelos reales usando `test_native_engine.py`
- [ ] Verificar que funciona en producción (cargar desde frontend)
- [ ] Optimizar conversión disperso ↔ denso
- [ ] Crear benchmarks y documentar resultados
- [ ] Resolver cualquier issue de compatibilidad
- [ ] Documentar diferencias de comportamiento (si las hay)

---

## 🔗 Referencias

- [[PHASE_2_SETUP_LOG]]: Log de setup inicial
- [[PHASE_2_NATIVE_ENGINE_TEST]]: Documentación del script de test
- `src/engines/native_engine_wrapper.py`: Wrapper Python
- `src/cpp_core/`: Código fuente C++

---

**Estado:** ⏳ Migración en progreso  
**Próximo paso:** Verificar compilación y probar con modelos reales

