---
title: Implementación de Visualizaciones con Shaders (GPU)
type: experiment
status: in_progress
tags: [optimization, visualization, performance, gpu, webgl, shaders]
created: 2025-11-21
updated: 2025-11-21
related: [[40_Experiments/VISUALIZATION_OPTIMIZATION_ANALYSIS|Análisis de Optimización de Visualización]], [[30_Components/Native_Engine_Core|Motor Nativo C++]]
---

# 🚀 Implementación de Visualizaciones con Shaders (GPU)

**Fecha**: 2025-11-21  
**Objetivo**: Implementar visualizaciones procesadas en GPU del navegador usando shaders WebGL para reducir el overhead del backend y mejorar el rendimiento.

---

## 📊 Contexto

Según el análisis en `VISUALIZATION_OPTIMIZATION_ANALYSIS.md`, la **Opción 3 (Shaders en Frontend)** fue identificada como una estrategia híbrida recomendada para reducir el procesamiento en el backend y mejorar el rendimiento de visualización.

### Flujo Actual vs. Optimizado

**Antes (CPU en Backend)**:
```
Motor C++ → Python (GPU) → Cálculos visualización → numpy arrays → WebSocket → Frontend (Canvas2D)
```

**Después (GPU en Frontend)**:
```
Motor C++ → Python → Datos raw (psi/map_data) → WebSocket → Frontend (WebGL Shaders) → GPU navegador
```

---

## ✨ Implementación

### 1. Sistema de Shaders WebGL (`shaderVisualization.ts`)

Utilidades para detectar y usar WebGL/WebGL2, compilar shaders, y crear texturas desde datos 2D.

#### Componentes Principales

- **Detección de WebGL**: `isWebGLAvailable()`, `isWebGL2Available()`
- **Shaders**:
  - `VERTEX_SHADER_2D`: Shader de vertex básico para renderizado 2D
  - `FRAGMENT_SHADER_DENSITY`: Shader para visualización de densidad (|ψ|²)
  - `FRAGMENT_SHADER_PHASE`: Shader para visualización de fase (angle(ψ))
- **Colormaps en GPU**: Viridis, Plasma implementados directamente en shaders
- **Utilidades**: `createShaderProgram()`, `createTextureFromData()`, `renderWithShader()`

#### Características

- **Compatibilidad**: Soporta WebGL1 y WebGL2 (fallback automático)
- **Precisión**: Normalización automática de valores para máxima compatibilidad
- **Colormaps**: Implementados en GPU (Viridis, Plasma, grayscale)
- **Configuración**: Min/max values, gamma correction, colormap selection

### 2. Componente ShaderCanvas (`ShaderCanvas.tsx`)

Componente React que usa WebGL para renderizar visualizaciones con shaders.

#### Funcionalidades

- **Detección Automática**: Detecta WebGL y usa shaders si está disponible
- **Fallback**: Retorna `null` si WebGL no está disponible (padre usa Canvas2D)
- **Renderizado Automático**: Actualiza cuando cambian `mapData`, `width`, `height`, `selectedViz`
- **Normalización Automática**: Calcula min/max automáticamente si no se proporcionan

#### Props

```typescript
interface ShaderCanvasProps {
    mapData: number[][];
    width: number;
    height: number;
    selectedViz: string;
    minValue?: number;
    maxValue?: number;
    className?: string;
    style?: React.CSSProperties;
}
```

---

## 🎯 Beneficios Esperados

### Rendimiento

| Métrica | Canvas2D (CPU) | WebGL Shaders (GPU) | Mejora |
| :------ | :-------------- | :------------------ | :----- |
| **Renderizado 256x256** | ~16ms (pixel loop) | ~1-2ms (shader) | **8-16x más rápido** |
| **Renderizado 512x512** | ~64ms (pixel loop) | ~2-4ms (shader) | **16-32x más rápido** |
| **Renderizado 1024x1024** | ~256ms (pixel loop) | ~4-8ms (shader) | **32-64x más rápido** |
| **Uso CPU** | Alto (loop pixel) | Bajo (GPU) | **Reducción significativa** |
| **Uso GPU** | N/A | Alto (navegador) | **Procesamiento paralelo** |

### Backend

- **Menos Procesamiento**: Puede enviar datos raw (psi) sin procesar visualizaciones
- **Menos Transferencia**: Opcionalmente enviar solo datos esenciales (map_data minimizado)
- **Mejor Escalabilidad**: Backend puede enfocarse en simulación, frontend en visualización

---

## 🔧 Próximos Pasos

### Fase 1: Integración en PanZoomCanvas ✅ (En Progreso)

- [ ] Integrar `ShaderCanvas` en `PanZoomCanvas` como alternativa cuando WebGL está disponible
- [ ] Mantener fallback a Canvas2D para compatibilidad
- [ ] Asegurar que overlays y pan/zoom funcionen correctamente

### Fase 2: Optimización de Pipeline Backend

- [ ] Modificar `pipeline_viz.py` para detectar si frontend soporta shaders
- [ ] Enviar datos raw (psi) cuando shaders están disponibles
- [ ] Mantener procesamiento actual para compatibilidad con Canvas2D

### Fase 3: Shaders Avanzados

- [ ] Implementar shader para visualización de fase (phase_hsv)
- [ ] Implementar shader para visualización de flujo (flow/quiver)
- [ ] Agregar más colormaps (inferno, magma, turbo)

### Fase 4: Documentación

- [ ] Documentar API de shaders
- [ ] Crear guía de uso
- [ ] Documentar beneficios y métricas de rendimiento

---

## 🧪 Testing

### Verificación de WebGL

- [ ] Detectar WebGL en diferentes navegadores (Chrome, Firefox, Safari, Edge)
- [ ] Verificar fallback a Canvas2D cuando WebGL no está disponible
- [ ] Probar compatibilidad con WebGL1 y WebGL2

### Rendimiento

- [ ] Benchmark renderizado Canvas2D vs. WebGL Shaders
- [ ] Medir uso de CPU/GPU en diferentes tamaños de grid
- [ ] Verificar escalabilidad para grids grandes (1024x1024+)

### Funcionalidad

- [ ] Verificar que pan/zoom funciona correctamente con shaders
- [ ] Verificar que overlays se renderizan correctamente
- [ ] Verificar que diferentes tipos de visualización funcionan (density, phase, etc.)

---

## 📈 Métricas Actuales

### Implementación Actual

- **Sistema de Shaders**: ✅ Implementado (`shaderVisualization.ts`)
- **Componente ShaderCanvas**: ✅ Implementado (`ShaderCanvas.tsx`)
- **Integración en PanZoomCanvas**: ⏳ En Progreso
- **Optimización Backend**: ⏳ Pendiente

### Próximas Mediciones

- Tiempo de renderizado por frame (256x256, 512x512, 1024x1024)
- Uso de CPU/GPU durante renderizado
- Comparación con implementación Canvas2D actual

---

## 🔗 Referencias

- [[40_Experiments/VISUALIZATION_OPTIMIZATION_ANALYSIS|Análisis de Optimización de Visualización]] - Análisis completo del flujo actual
- [[30_Components/Native_Engine_Core|Motor Nativo C++]] - Motor de simulación
- `src/pipelines/pipeline_viz.py` - Pipeline de visualización actual
- `frontend/src/components/ui/PanZoomCanvas.tsx` - Componente de canvas actual
- `frontend/src/utils/shaderVisualization.ts` - Sistema de shaders
- `frontend/src/components/ui/ShaderCanvas.tsx` - Componente de canvas con shaders

---

*Última actualización: 2025-11-21*

