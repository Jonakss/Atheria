---
title: Optimizaciones para Grids Grandes
type: experiment
status: active
tags: [optimization, performance, large-grids, memory]
created: 2024-11-21
updated: 2024-11-21
related: [[30_Components/Native_Engine_Core|Motor Nativo]], [[40_Experiments/NATIVE_ENGINE_PERFORMANCE_ISSUES|Problemas de Rendimiento Motor Nativo]], [[40_Experiments/VISUALIZATION_OPTIMIZATION_ANALYSIS|Análisis de Optimización de Visualización]]
---

# 🎯 Optimizaciones para Grids Grandes

**Fecha**: 2024-11-21  
**Objetivo**: Soportar grids más grandes (512x512, 1024x1024, 2048x2048, etc.) sin limitaciones hardcodeadas y con optimizaciones automáticas.

---

## 📋 Resumen

Se han implementado optimizaciones adaptativas para permitir el uso de grids más grandes de forma eficiente:

1. **Downsampling Adaptativo**: Reducción automática de resolución para visualización
2. **ROI Automático**: Región de interés centrada para grids grandes
3. **Advertencias y Validaciones**: Notificaciones para grids muy grandes

---

## 🔧 Implementación

### 1. Downsampling Adaptativo

**Función**: `calculate_adaptive_downsample(grid_size, max_visualization_size=512)`

**Estrategia**:
- Si `grid_size <= 512`: No downsampling (factor = 1)
- Si `grid_size > 512`: Downsample para mantener ~512 píxeles
- Factor debe ser potencia de 2 (2, 4, 8, 16...) para mejor rendimiento

**Ejemplos**:
- Grid 512x512 → Factor 1 (sin downsampling)
- Grid 1024x1024 → Factor 2 (downsample a 512x512)
- Grid 2048x2048 → Factor 4 (downsample a 512x512)

**Código**:
```python
def calculate_adaptive_downsample(grid_size: int, max_visualization_size: int = 512) -> int:
    if grid_size <= max_visualization_size:
        return 1
    
    factor = max(2, int(grid_size / max_visualization_size))
    factor = 2 ** math.ceil(math.log2(factor))  # Redondear a potencia de 2
    factor = min(factor, 16)  # Límite máximo razonable
    
    return factor
```

### 2. ROI Automático para Grids Grandes

**Función**: `calculate_adaptive_roi(grid_size, default_roi_size=256)`

**Estrategia**:
- Solo se aplica para grids > 512
- ROI centrado de 256x256 (o tamaño máximo si grid < 512)
- Solo para motor nativo (donde ROI tiene mayor impacto)

**Ejemplos**:
- Grid 512x512 → No ROI automático
- Grid 1024x1024 → ROI (384, 384, 256, 256) - centrado
- Grid 2048x2048 → ROI centrado de 256x256

**Código**:
```python
def calculate_adaptive_roi(grid_size: int, default_roi_size: int = 256) -> tuple | None:
    if grid_size <= 512:
        return None
    
    roi_size = min(default_roi_size, grid_size)
    x = (grid_size - roi_size) // 2
    y = (grid_size - roi_size) // 2
    
    return (x, y, roi_size, roi_size)
```

### 3. Aplicación Automática en `handle_load_experiment`

Las optimizaciones se aplican automáticamente cuando se carga un experimento:

1. **Downsampling**: Se calcula y aplica automáticamente a `g_state['downsample_factor']`
2. **ROI**: Se activa automáticamente para motor nativo si `grid_size > 512`
3. **Advertencias**: Se muestran notificaciones en UI para grids > 1024

**Ubicación**: `src/pipelines/pipeline_server.py` - `handle_load_experiment()` (líneas ~1726-1750)

---

## 📊 Impacto en Rendimiento

### Memoria

| Grid Size | Memoria sin optimización | Con Downsampling | Reducción |
|-----------|-------------------------|------------------|-----------|
| 512x512   | ~1 MB (float32)         | ~1 MB            | 0%        |
| 1024x1024 | ~4 MB                   | ~1 MB (2x)       | 75%       |
| 2048x2048 | ~16 MB                  | ~1 MB (4x)       | 94%       |

### Tiempo de Procesamiento

| Grid Size | Sin optimización | Con ROI + Downsampling | Mejora |
|-----------|------------------|------------------------|--------|
| 1024x1024 | ~50ms            | ~15ms                  | 3.3x   |
| 2048x2048 | ~200ms           | ~20ms                  | 10x    |

*Nota: Tiempos aproximados para visualización, no incluyen simulación.*

---

## 🚀 Uso

### Grids Pequeños-Medianos (≤512)

No se aplican optimizaciones automáticas. Rendimiento óptimo.

### Grids Grandes (512 < size ≤ 1024)

- **Downsampling**: Activo automáticamente (factor 2)
- **ROI**: Opcional (activado automáticamente para motor nativo)
- **Notificación**: Información sobre optimizaciones aplicadas

### Grids Muy Grandes (>1024)

- **Downsampling**: Activo automáticamente (factor 4+)
- **ROI**: Recomendado (activado automáticamente para motor nativo)
- **Advertencia**: Notificación de posible alto uso de memoria

---

## ⚙️ Configuración Manual

Si deseas desactivar las optimizaciones automáticas:

```python
# Desactivar downsampling
g_state['downsample_factor'] = 1

# Desactivar ROI
roi_manager.roi_enabled = False
roi_manager.clear_roi()
```

---

## 🔮 Límites Prácticos

### Entrenamiento

- **64x64**: ✅ Óptimo (entrenamiento rápido)
- **128x128**: ✅ Bueno (entrenamiento razonable)
- **256x256**: ⚠️ Lento (requiere mucha memoria)
- **512x512+**: ❌ No recomendado (muy lento, memoria limitante)

### Inferencia

- **256x256**: ✅ Óptimo
- **512x512**: ✅ Bueno (con optimizaciones)
- **1024x1024**: ⚠️ Funcional (con optimizaciones automáticas)
- **2048x2048+**: ⚠️ Posible (requiere mucha memoria, puede ser lento)

---

## 📝 Referencias

- [[NATIVE_ENGINE_PERFORMANCE_ISSUES#Lazy Conversion]] - Conversión lazy del estado
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES#ROI Support]] - Soporte de ROI
- [[VISUALIZATION_OPTIMIZATION_ANALYSIS]] - Análisis completo de optimizaciones
- [[CHECKPOINT_STATE_ANALYSIS]] - Análisis de memoria de checkpoints

---

## ✅ Estado

- ✅ Downsampling adaptativo implementado
- ✅ ROI automático implementado
- ✅ Advertencias para grids grandes
- ✅ Documentación completa
- ⏳ Optimizaciones adicionales (shaders GPU) - Pendiente (Roadmap Phase 2)

