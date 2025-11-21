---
title: Error CUDA Motor Nativo - undefined symbol __nvJitLinkCreate
type: issue
status: open
tags: [cuda, native-engine, troubleshooting]
created: 2025-11-20
updated: 2025-11-20
related: [[NATIVE_ENGINE_PERFORMANCE_ISSUES|Problemas Motor Nativo]], [[Native_Engine_Core|Motor Nativo]]
---

# Error CUDA Motor Nativo: undefined symbol __nvJitLinkCreate

**Fecha**: 2025-11-20  
**Estado**: 🔴 **ABIERTO** - Problema de configuración de sistema  
**Prioridad**: 🟡 Media (el motor Python funciona como fallback)

---

## 🐛 Error Observado

```
ImportError: /home/jonathan.correa/Projects/Atheria/ath_venv/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12
```

**Cuándo ocurre:**
- Al intentar importar `atheria_core` desde Python
- Específicamente cuando PyTorch intenta cargar las librerías de CUDA
- El motor nativo C++ depende de LibTorch, que a su vez depende de librerías CUDA

---

## 🔍 Causa Raíz

Este error indica que hay un **conflicto de versiones entre las librerías de CUDA**:

1. **PyTorch** compilado con una versión de CUDA (ej: CUDA 12.8)
2. **Sistema** tiene instaladas librerías de CUDA de una versión diferente
3. La librería `libnvJitLink.so.12` no puede encontrar el símbolo `__nvJitLinkCreate_12_8`

**Símbolo faltante:**
- `__nvJitLinkCreate_12_8`: Parte de CUDA JIT Linker API (versión 12.8)
- Indica que PyTorch espera CUDA 12.8, pero el sistema puede tener otra versión

---

## 🛠️ Soluciones Propuestas

### Solución 1: Verificar Versión de CUDA

```bash
# Verificar versión de CUDA instalada en el sistema
nvcc --version

# Verificar versión que PyTorch espera
python3 -c "import torch; print(torch.version.cuda)"

# Si no coinciden, actualizar PyTorch o CUDA
```

### Solución 2: Usar Motor Python (Temporal)

El motor Python funciona correctamente como fallback:

```python
# En pipeline_server.py, el motor Python se usa automáticamente
# cuando el motor nativo no está disponible
use_native_engine = False  # Forzar uso de motor Python
```

### Solución 3: Compilar Motor Nativo Solo para CPU

Si CUDA tiene problemas, compilar el motor nativo para CPU mode:

```bash
# En CMakeLists.txt, forzar CPU mode
cmake -DTORCH_CUDA=OFF ...

# O usar device='cpu' al inicializar
motor = NativeEngineWrapper(..., device='cpu')
```

### Solución 4: Actualizar PyTorch/CUDA Toolkit

```bash
# Reinstalar PyTorch con la versión correcta de CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# O usar CUDA toolkit compatible
conda install cudatoolkit=12.1 -c pytorch
```

---

## 📊 Impacto

**Funcionalidad:**
- ✅ Motor Python: **Funciona correctamente** (fallback automático)
- ❌ Motor Nativo: **No disponible** hasta resolver problema CUDA

**Rendimiento:**
- Motor Python: ~100-500 steps/segundo (depende del modelo)
- Motor Nativo: ~10,000 steps/segundo (cuando funciona)

**Trabajo Actual:**
- Las optimizaciones de tiempo real (lazy conversion, ROI) ya implementadas
- El motor Python se beneficia de estas optimizaciones también
- **No bloquea desarrollo** de optimizaciones adicionales

---

## 🔗 Referencias

- [[NATIVE_ENGINE_PERFORMANCE_ISSUES|Problemas Motor Nativo]] - Optimizaciones ya implementadas
- [[Native_Engine_Core|Motor Nativo]] - Documentación del motor nativo
- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/) - Guía de compatibilidad

---

## 📝 Notas Adicionales

**Workaround Actual:**
El sistema detecta automáticamente cuando el motor nativo no está disponible y usa el motor Python como fallback. Esto permite que el desarrollo continúe sin bloqueos.

**Próximos Pasos:**
1. Verificar versión de CUDA en el sistema
2. Actualizar PyTorch o CUDA toolkit si es necesario
3. Compilar motor nativo con la versión correcta de CUDA
4. Probar importación de `atheria_core`

**Optimizaciones Futuras:**
- Las optimizaciones de tiempo real (paralelismo, SIMD, visualización C++) se pueden implementar independientemente
- El motor Python también se beneficiará de estas optimizaciones cuando se implementen

---

**Última actualización:** 2025-11-20

