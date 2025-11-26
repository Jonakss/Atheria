---
title: Mejoras del Visor 3D - Zoom, Pan y Tamaño
type: improvement
status: in-progress
tags: [frontend, 3d-viewer, ux, zoom, pan]
created: 2025-11-21
updated: 2025-11-21
related: [[30_Components/HolographicViewer|Visor Holográfico]], [[VISUALIZATION_OPTIMIZATION_ANALYSIS|Análisis de Optimización de Visualización]]
---

# Mejoras del Visor 3D - Zoom, Pan y Tamaño

**Fecha**: 2025-11-21  
**Estado**: 🔄 En Progreso  
**Prioridad**: 🟡 Alta

---

## 🐛 Problemas Identificados

### 1. Redimensionamiento Afecta Zoom/Pan
**Problema:**
- Cuando se redimensiona la ventana, el visor recalcula automáticamente el zoom/pan
- Esto cambia la vista que el usuario tenía configurada
- Frustrante para el usuario que ajustó manualmente la vista

**Causa:**
- `usePanZoom` recalcula `calculateInitialView()` cuando cambia el tamaño del contenedor
- El visor usa `width: 100%, height: 100%` dependiendo del tamaño de la ventana

**Solución Propuesta:**
- Agregar listener de resize que NO recalcule zoom/pan automáticamente
- Solo ajustar tamaño del renderer sin cambiar la vista
- Mantener zoom/pan del usuario incluso al redimensionar

### 2. Visor Ocupa Toda la Ventana
**Problema:**
- El visor siempre ocupa `width: 100%, height: 100%`
- No se puede hacer más pequeño que la ventana
- Todo cambia cuando se redimensiona la ventana

**Causa:**
- Estilos fijos: `style={{ width: '100%', height: '100%', minHeight: '400px' }}`
- No hay controles para ajustar el tamaño del visor

**Solución Propuesta:**
- Agregar controles de tamaño (slider o botones)
- Permitir tamaño fijo independiente de la ventana
- Guardar preferencias de tamaño en localStorage

### 3. Zoom/Pan "Raro"
**Problema:**
- El zoom y pan no se sienten naturales
- Puede haber problemas con la velocidad o la sensibilidad

**Causa:**
- Lógica compleja de transformación CSS con `transform: scale() translate()`
- Cálculos de límites que pueden ser confusos
- No hay feedback visual claro del zoom/pan

**Solución Propuesta:**
- Revisar y simplificar la lógica de zoom/pan
- Ajustar sensibilidad del mouse/rueda
- Agregar indicadores visuales de zoom (nivel de zoom visible)

### 4. ROI con Zoom Causa Distorsión
**Problema:**
- Cuando se aplica ROI en el backend y luego se hace zoom en el frontend
- Los datos ya están procesados (pixelados/distorsionados)
- Mejor sería usar shaders para zoom suave

**Causa:**
- ROI se aplica en backend (reduciendo datos)
- Zoom se hace en frontend sobre datos ya procesados
- No hay interpolación suave

**Solución Propuesta (Futuro):**
- **Shaders WebGL** para procesar ROI/zoom en GPU
- Enviar datos completos desde backend
- Aplicar ROI/zoom con shaders para zoom suave sin distorsión
- Interpolación bilinear/bicúbica en shader

---

## ✅ Soluciones Implementadas

### 1. Mejora de Resize (✅ Implementado)
- [x] Listener de resize que NO recalcule zoom/pan
- [x] Solo ajustar tamaño del renderer y aspect ratio de la cámara
- [x] Mantener vista del usuario (posición de cámara y controles) al redimensionar
- [x] Usar ResizeObserver para detectar cambios de tamaño del contenedor

### 2. Controles de Tamaño (Pendiente)
- [ ] Slider o botones para tamaño del visor
- [ ] Guardar preferencias en localStorage
- [ ] Tamaño fijo independiente de ventana

### 3. Mejora de Zoom/Pan (Pendiente)
- [ ] Revisar lógica de transformaciones
- [ ] Ajustar sensibilidad
- [ ] Indicadores visuales de zoom

### 4. Shaders para ROI/Zoom (Futuro)
- [ ] Documentar necesidad
- [ ] Evaluar implementación con WebGL shaders
- [ ] Planificar migración gradual

---

## 📋 Plan de Implementación

### Fase 1: Correcciones Inmediatas (Alta Prioridad)
1. **Fix Resize**: Modificar `usePanZoom` para que NO recalcule al resize
2. **Tamaño Fijo**: Agregar controles básicos de tamaño
3. **Mejorar Zoom/Pan**: Ajustar sensibilidad y lógica

### Fase 2: Mejoras UX (Media Prioridad)
1. **Indicadores Visuales**: Mostrar nivel de zoom
2. **Controles de Vista**: Botones para reset, fit, etc.
3. **Guardar Preferencias**: Tamaño y vista en localStorage

### Fase 3: Optimizaciones Futuras (Baja Prioridad)
1. **Shaders WebGL**: Procesar ROI/zoom en GPU
2. **Interpolación**: Zoom suave con shaders
3. **Performance**: Optimizar renderizado para grandes datasets

---

## 🔗 Referencias

- [[30_Components/HolographicViewer|Visor Holográfico]] - Componente actual
- [[VISUALIZATION_OPTIMIZATION_ANALYSIS|Análisis de Optimización]] - Opciones de optimización
- `frontend/src/hooks/usePanZoom.ts` - Hook de zoom/pan actual
- `frontend/src/components/visualization/HolographicViewer.tsx` - Visor 3D actual

---

**Última actualización**: 2025-11-21

