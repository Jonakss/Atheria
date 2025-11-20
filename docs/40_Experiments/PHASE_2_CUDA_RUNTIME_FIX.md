# 🔧 Fix de CUDA Runtime para Motor Nativo

**Fecha:** 2024-12-XX  
**Problema:** Error al importar `atheria_core` debido a problema de CUDA runtime.

---

## 🐛 Problema Identificado

**Error:**
```
ImportError: /home/jonathan.correa/Projects/Atheria/ath_venv/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12
```

**Causa:**
- El módulo C++ está compilado correctamente
- El problema es un error de runtime CUDA (conflicto de versiones de librerías)
- El símbolo `__nvJitLinkCreate_12_8` no está disponible en la versión de `libnvJitLink.so.12`
- Esto ocurre al intentar cargar las librerías CUDA al importar el módulo

**Impacto:**
- El módulo no se puede importar si CUDA está habilitado
- El motor nativo no puede inicializarse
- Fallback automático a motor Python funciona correctamente

---

## ✅ Solución Implementada

### 1. Detección Mejorada de Errores CUDA

**Archivo:** `src/engines/native_engine_wrapper.py`

**Cambios:**
- Detección específica de errores de CUDA runtime
- Flag `_native_cuda_issue` para indicar problemas de CUDA
- Manejo diferenciado de errores de CUDA vs errores de compilación

```python
# Detectar problemas específicos de CUDA runtime
cuda_runtime_keywords = [
    '__nvJitLinkCreate',
    'libnvJitLink',
    'libcusparse.so',
    'undefined symbol'
]
```

### 2. Importación Forzando CPU Mode

**Estrategia:**
- Si hay problema de CUDA pero se intenta usar CPU, deshabilitar CUDA temporalmente
- Establecer `CUDA_VISIBLE_DEVICES=''` antes de importar
- Restaurar valor original después de importar

```python
if not NATIVE_AVAILABLE and _native_cuda_issue and device == "cpu":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import atheria_core  # Reintentar importación en CPU mode
```

### 3. Fallback Automático a CPU

**Comportamiento:**
- Si se intenta inicializar con `device="cuda"` pero hay problema de CUDA
- Automáticamente cambiar a `device="cpu"` con warning
- Permitir que el motor nativo funcione en CPU mode

---

## 🔍 Verificación

### Test de Importación

```python
# Probar importación directa en CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import atheria_core

# Crear Engine en CPU mode
engine = atheria_core.Engine(d_state=8, device_str='cpu')
```

### Test con Wrapper

```python
from src.engines.native_engine_wrapper import NativeEngineWrapper

# Intentar inicializar con CPU (debería funcionar)
wrapper = NativeEngineWrapper(
    grid_size=128,
    d_state=8,
    device='cpu'  # Forzar CPU mode
)
```

---

## 📝 Estado Actual

✅ **Completado:**
- Detección mejorada de errores CUDA runtime
- Importación forzando CPU mode implementada
- Fallback automático a CPU cuando hay problemas de CUDA
- Manejo robusto de errores en wrapper

⏳ **Pendiente:**
- Resolver problema de CUDA runtime a nivel de librerías (opcional)
- Verificar que funciona con modelos reales
- Optimizar conversión disperso ↔ denso

---

## 🔗 Referencias

- [[PHASE_2_MIGRATION_TO_NATIVE]]: Guía completa de migración
- `src/engines/native_engine_wrapper.py`: Implementación del wrapper

---

**Estado:** ✅ Fix implementado - Motor nativo funciona en CPU mode  
**Próximo paso:** Probar con modelos reales

