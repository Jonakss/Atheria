# 🧪 Test de Motor Nativo con Modelos Reales

**Fecha:** 2024-12-XX  
**Objetivo:** Probar el motor nativo C++ con modelos PyTorch entrenados reales.

---

## 📋 Resumen

Este documento describe el script de prueba `scripts/test_native_engine.py` que permite:

1. **Cargar modelos entrenados** desde checkpoints
2. **Exportar automáticamente** a TorchScript
3. **Inicializar el motor nativo C++** con el modelo
4. **Ejecutar pasos de simulación** y medir rendimiento
5. **Verificar estado cuántico** y métricas

---

## 🚀 Uso

### Ejecución Básica

```bash
python scripts/test_native_engine.py --experiment UNET_32ch_D5_LR2e-5
```

### Opciones

```bash
python scripts/test_native_engine.py \
    --experiment UNET_32ch_D5_LR2e-5 \
    --device cpu \
    --steps 10
```

**Parámetros:**
- `--experiment`: Nombre del experimento a probar (requerido)
- `--device`: Dispositivo (`cpu` o `cuda`, por defecto `cpu`)
- `--steps`: Número de pasos de simulación (por defecto `10`)

---

## 🔄 Flujo de Ejecución

### 1. Verificación de Módulo C++

El script verifica que `atheria_core` esté disponible:

```python
import atheria_core
assert atheria_core.has_torch_support()
```

### 2. Carga de Configuración

Carga la configuración del experimento desde `output/experiments/{experiment_name}/config.json`:

- Arquitectura del modelo
- Parámetros (`d_state`, `hidden_channels`, etc.)
- Grid size para inferencia

### 3. Carga de Modelo Entrenado

Carga el checkpoint más reciente usando `get_latest_checkpoint()`:

```python
checkpoint_path = get_latest_checkpoint(experiment_name)
model, state_dict = load_model(exp_config, checkpoint_path)
```

### 4. Exportación a TorchScript

Exporta el modelo a formato TorchScript (`.pt`):

```python
exported_path = export_model_to_torchscript(
    model, 
    device, 
    output_path,
    grid_size=grid_size,
    d_state=d_state
)
```

**Estrategia de Exportación:**
- Intenta `torch.jit.script()` primero (más optimizado)
- Fallback a `torch.jit.trace()` si falla (más compatible)
- Guarda en `output/torchscript_models/{experiment_name}.pt`

### 5. Inicialización del Motor Nativo

Crea una instancia de `NativeEngineWrapper`:

```python
wrapper = NativeEngineWrapper(
    grid_size=grid_size,
    d_state=d_state,
    device=device_str,
    cfg=exp_config
)
```

### 6. Carga del Modelo en C++

Carga el modelo TorchScript en el motor nativo:

```python
success = wrapper.load_model(str(torchscript_path))
```

### 7. Agregar Partículas Iniciales

Agrega partículas iniciales para la simulación:

```python
wrapper.add_initial_particles(num_particles=10)
```

### 8. Ejecución de Simulación

Ejecuta pasos de simulación y mide rendimiento:

```python
for step in range(num_steps):
    start_time = time.time()
    wrapper.evolve_internal_state()
    elapsed = time.time() - start_time
    # ... registrar métricas
```

### 9. Verificación de Estado Final

Verifica que el estado cuántico sea válido:

- Shape del tensor `psi`
- Device y dtype
- Estadísticas (min, max, mean)

---

## 📊 Métricas Reportadas

El script reporta:

1. **Tiempo promedio por paso** (ms)
2. **Tiempo total** (s)
3. **Partículas promedio** durante la simulación
4. **Throughput** (pasos/segundo)
5. **Estadísticas del estado cuántico** (min, max, mean)

---

## 🔍 Verificaciones

### Estado Cuántico

- ✅ Tensor `psi` no es `None`
- ✅ Shape correcto: `[1, H, W, d_state]`
- ✅ Device correcto (CPU o CUDA)
- ✅ Dtype: `torch.complex64`
- ✅ Valores finitos (no NaN, no Inf)

### Rendimiento

- ✅ Tiempo por paso < 100ms (objetivo)
- ✅ Throughput > 10 pasos/segundo (objetivo)
- ✅ Sin errores de memoria o CUDA

---

## 🐛 Troubleshooting

### Error: "atheria_core no disponible"

**Solución:** Compilar el módulo C++:

```bash
python setup.py build_ext --inplace
```

### Error: "No se encontró checkpoint"

**Solución:** Asegurarse de que el experimento existe y tiene checkpoints:

```bash
ls output/training_checkpoints/{experiment_name}/
```

### Error: "Error exportando modelo"

**Causas posibles:**
- Modelo no compatible con TorchScript
- Operaciones dinámicas no soportadas
- Problemas de device (CPU vs CUDA)

**Solución:** Verificar que el modelo use operaciones estáticas y sea compatible con JIT.

### Error: "Error cargando modelo en motor nativo"

**Causas posibles:**
- Modelo TorchScript corrupto
- Incompatibilidad de versiones LibTorch
- Problemas de CUDA runtime

**Solución:** 
- Verificar que el modelo se exportó correctamente
- Probar con `--device cpu` primero
- Verificar versiones de PyTorch/LibTorch

---

## 📝 Ejemplo de Salida

```
================================================================================
🧪 TEST: Motor Nativo C++ con Modelo Real
================================================================================

✅ Módulo C++ importable: atheria_core
   has_torch_support: True

📋 Cargando configuración del experimento: UNET_32ch_D5_LR2e-5
✅ Configuración cargada
   Arquitectura: UNET_UNITARY
   d_state: 8
   Grid size (inference): 128

📦 Cargando modelo entrenado...
   Checkpoint: output/training_checkpoints/UNET_32ch_D5_LR2e-5/qca_checkpoint_eps195.pth
✅ Modelo cargado exitosamente
   Tipo: UNetUnitary

📤 Exportando modelo a TorchScript...
  Input shape: torch.Size([1, 16, 128, 128])
  Device: cpu
  Intentando torch.jit.script...
✅ Modelo exportado a: output/torchscript_models/UNET_32ch_D5_LR2e-5.pt
✅ Modelo TorchScript verificado (carga exitosa)

🚀 Inicializando motor nativo C++...
✅ Motor nativo inicializado

📥 Cargando modelo TorchScript en motor nativo...
✅ Modelo cargado en motor nativo

✨ Agregando partículas iniciales...
✅ 10 partículas agregadas

⏱️  Ejecutando 10 pasos de simulación...
   Paso 1/10: 45.23ms, 10 partículas, step_count=1
   Paso 2/10: 42.15ms, 10 partículas, step_count=2
   ...
   Paso 10/10: 43.67ms, 10 partículas, step_count=10

📊 Métricas de Rendimiento:
   Tiempo promedio por paso: 43.52ms
   Tiempo total: 0.435s
   Partículas promedio: 10.0
   Throughput: 22.99 pasos/segundo

🔍 Verificando estado final...
✅ Estado cuántico disponible
   Shape: torch.Size([1, 128, 128, 8])
   Device: cpu
   Dtype: torch.complex64
   Es complejo: True
   Min: 0.000123
   Max: 0.456789
   Mean: 0.012345

================================================================================
✅ TEST COMPLETADO EXITOSAMENTE
================================================================================
```

---

## 🔗 Referencias

- [[PHASE_2_SETUP_LOG]]: Log de setup de Phase 2
- [[Native_Engine_Core]]: Documentación del motor nativo
- [PyTorch JIT Documentation](https://pytorch.org/docs/stable/jit.html)
- [LibTorch C++ API](https://pytorch.org/cppdocs/)

---

**Estado:** ✅ Script implementado y listo para pruebas  
**Siguiente:** Ejecutar tests con modelos reales y comparar rendimiento
