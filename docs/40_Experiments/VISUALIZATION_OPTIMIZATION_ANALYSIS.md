---
title: Análisis de Optimización de Visualización
type: experiment
status: active
tags: [optimization, visualization, performance, cpp, gpu]
created: 2025-11-20
updated: 2025-11-20
related: [[30_Components/Native_Engine_Core|Motor Nativo]], [[30_Components/WEB_SOCKET_PROTOCOL|Protocolo WebSocket]], [[40_Experiments/ARCHITECTURE_EVALUATION_GO_VS_PYTHON|Evaluación Go vs Python]]
---

# 🔍 Análisis de Optimización de Visualización

**Fecha**: 2025-11-20  
**Objetivo**: Analizar quién hace la visualización y cómo optimizar el envío de datos.

---

## 📊 Flujo Actual de Visualización

### Estado Actual

```
┌─────────────────┐
│ Motor Nativo C++│
│ (Estado Disperso)│
└────────┬────────┘
         │ get_dense_state()
         ▼
┌─────────────────┐
│ Python Wrapper  │
│ Conversión      │
│ Sparse → Dense  │
└────────┬────────┘
         │ psi (torch.Tensor denso)
         ▼
┌─────────────────┐
│ pipeline_viz.py │
│ Python (GPU)    │
│ - density = |ψ|²│
│ - phase = angle │
│ - energy, etc.  │
└────────┬────────┘
         │ numpy arrays
         ▼
┌─────────────────┐
│ WebSocket       │
│ MessagePack     │
│ (Binario)       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Frontend        │
│ React/Three.js  │
│ Renderizado     │
└─────────────────┘
```

### Componentes Actuales

1. **Motor Nativo C++**: Genera estado disperso (solo partículas activas)
2. **Python Wrapper**: Convierte disperso → denso (lazy conversion, ROI support)
3. **pipeline_viz.py (Python)**: Calcula visualizaciones en GPU
   - `get_visualization_data()` procesa `psi` (tensor denso)
   - Cálculos vectorizados en CUDA: `|ψ|²`, `angle(ψ)`, etc.
   - Conversión a numpy arrays para serialización
4. **WebSocket**: Serializa con MessagePack (binario)
5. **Frontend**: Recibe y renderiza

---

## ⚠️ Cuellos de Botella Identificados

### 1. **Conversión Disperso → Denso (Python)**
- **Ubicación**: `native_engine_wrapper.py` → `_update_dense_state_from_sparse()`
- **Problema**: Iteración sobre coordenadas con llamadas Python↔C++
- **Optimización actual**: Lazy conversion + ROI (ya implementado)
- **Impacto**: ~0.1ms por frame (optimizado)

### 2. **Cálculos de Visualización (Python)**
- **Ubicación**: `pipeline_viz.py` → `get_visualization_data()`
- **Problema**: Cálculos en Python aunque vectorizados en GPU
- **Overhead**: Sincronización CUDA, conversión a numpy
- **Impacto**: ~2-5ms por frame

### 3. **Serialización y Transferencia**
- **Ubicación**: `server_state.py` → `broadcast()` → MessagePack
- **Problema**: Conversión numpy → MessagePack → bytes
- **Optimización actual**: MessagePack binario (3-5x más eficiente que JSON)
- **Impacto**: ~1-2ms por frame

### 4. **Renderizado Frontend**
- **Ubicación**: Frontend React/Three.js
- **Problema**: Procesamiento de arrays grandes en JavaScript
- **Impacto**: Variable (depende de visualización)

---

## 💡 Opciones de Optimización

### Opción 1: Visualización en C++ (Motor Nativo)

**Ventajas:**
- ✅ Elimina overhead Python
- ✅ Cálculos directos en GPU (LibTorch)
- ✅ Menos transferencias de memoria
- ✅ Paralelismo nativo (OpenMP/CUDA)

**Desventajas:**
- ❌ Requiere reimplementar lógica de visualización en C++
- ❌ Más complejidad en el motor nativo
- ❌ Mantenimiento de código duplicado

**Implementación:**
```cpp
// En Engine C++
torch::Tensor compute_visualization(
    const torch::Tensor& psi_dense,
    const std::string& viz_type
) {
    // Cálculos en GPU directamente
    auto density = psi_dense.abs().pow(2).sum(-1);
    auto phase = torch::angle(psi_dense);
    // ...
    return density; // o phase, energy, etc.
}
```

**Impacto esperado**: Reducción de ~2-5ms → ~0.5-1ms por frame

---

### Opción 2: Envío Directo desde GPU (Zero-Copy)

**Ventajas:**
- ✅ Evita transferencia GPU → CPU → WebSocket
- ✅ Datos permanecen en GPU hasta el último momento
- ✅ Menos copias de memoria

**Desventajas:**
- ❌ Requiere WebSocket con soporte GPU (WebGPU/WebGL)
- ❌ Frontend debe procesar datos binarios raw
- ❌ Más complejidad en el frontend

**Implementación:**
```python
# Enviar tensor directamente (sin convertir a numpy)
# Frontend recibe datos binarios raw y los procesa con shaders
binary_data = tensor_to_binary(psi_density)  # Directo desde GPU
```

**Impacto esperado**: Reducción de ~1-2ms → ~0.1-0.5ms por frame

---

### Opción 3: Shaders en Frontend (GPU Processing)

**Ventajas:**
- ✅ Cálculos de visualización en GPU del navegador
- ✅ Envío de datos raw (psi) sin procesar
- ✅ Renderizado eficiente con WebGL/WebGPU

**Desventajas:**
- ❌ Requiere reimplementar visualizaciones en shaders
- ❌ Más complejidad en el frontend
- ❌ Limitaciones de WebGL/WebGPU

**Implementación:**
```glsl
// Shader en frontend
uniform sampler2D psi_data;
void main() {
    vec4 psi = texture2D(psi_data, vUv);
    float density = dot(psi.rg, psi.rg) + dot(psi.ba, psi.ba);
    gl_FragColor = vec4(density, density, density, 1.0);
}
```

**Impacto esperado**: Reducción significativa en procesamiento frontend

---

### Opción 4: Híbrida (Recomendada)

**Estrategia:**
1. **C++ calcula visualizaciones básicas** (density, phase) en GPU
2. **Python solo para visualizaciones complejas** (Poincaré, t-SNE)
3. **Envío optimizado**: Datos raw cuando es posible, procesados cuando es necesario
4. **Frontend con shaders**: Para visualizaciones básicas (density, phase)

**Flujo Optimizado:**
```
Motor C++ → Cálculos básicos en GPU → Datos raw → WebSocket → Shaders Frontend
                ↓
         Visualizaciones complejas → Python → Procesado → WebSocket → Frontend
```

---

## 🎯 Recomendación

### Fase 1: Optimización Inmediata (Python)
- ✅ Ya implementado: Lazy conversion, ROI, MessagePack
- ⚠️ Pendiente: Optimizar sincronización CUDA en `pipeline_viz.py`

### Fase 2: Visualización en C++ (Corto Plazo)
- **Objetivo**: Mover cálculos básicos (density, phase) a C++
- **Impacto**: Reducción de ~2-5ms → ~0.5-1ms
- **Esfuerzo**: Medio (reimplementar en C++)

### Fase 3: Shaders en Frontend (Medio Plazo) ✅ **COMPLETADO**
- **Objetivo**: Procesar datos raw con shaders WebGL/WebGPU
- **Impacto**: Reducción significativa en procesamiento frontend
- **Esfuerzo**: Alto (reimplementar visualizaciones en shaders)
- **Estado**: ✅ Implementado y funcionando
  - ShaderCanvas integrado en PanZoomCanvas
  - Shaders implementados: density, phase, energy, real, imag
  - Detección automática de WebGL y uso condicional
  - Fallback a Canvas 2D para visualizaciones complejas

---

## 📈 Métricas Actuales

### Tiempos por Frame (256x256 grid)
- Conversión disperso→denso: ~0.1ms (optimizado)
- Cálculos visualización (Python/GPU): ~2-5ms
- Serialización MessagePack: ~1-2ms
- Transferencia WebSocket: ~0.5-1ms
- **Total**: ~4-8ms por frame

### Objetivo con Optimizaciones
- Visualización en C++: ~0.5-1ms
- Serialización optimizada: ~0.5-1ms
- Transferencia: ~0.5ms
- **Total objetivo**: ~1.5-2.5ms por frame (2-3x mejora)

---

## 🔗 Referencias

- [[30_Components/Native_Engine_Core|Motor Nativo C++]] - Arquitectura del motor
- [[30_Components/WEB_SOCKET_PROTOCOL|Protocolo WebSocket]] - Protocolo actual
- [[40_Experiments/ARCHITECTURE_EVALUATION_GO_VS_PYTHON|Evaluación Go vs Python]] - Análisis arquitectónico
- `src/pipelines/pipeline_viz.py` - Implementación actual
- `src/engines/native_engine_wrapper.py` - Conversión disperso→denso

---

*Última actualización: 2025-11-20*

