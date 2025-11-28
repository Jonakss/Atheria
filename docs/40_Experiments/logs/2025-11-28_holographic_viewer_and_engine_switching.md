# 2025-11-28 - Feature: Holographic Viewer & Engine Switching Docs

**Autor:** Antigravity Agent
**Tipo:** Feature / Documentation
**Estado:** Completado

## 📝 Resumen
Se implementó el **Visor Holográfico** (Poincaré Disk) utilizando shaders WebGL para visualizar la correspondencia AdS/CFT. Además, se creó documentación detallada sobre cómo cambiar entre los motores de simulación (Nativo vs Python).

## 🔧 Cambios Realizados

### 1. Visor Holográfico (Poincaré)
- **Nuevo Shader:** `frontend/src/shaders/poincare.frag` implementa la proyección conforme del grid 2D al disco hiperbólico.
- **Integración Frontend:**
    - Actualizado `ShaderCanvas.tsx` para incluir el nuevo shader.
    - Actualizado `PanZoomCanvas.tsx` para forzar el renderizado WebGL en modo `poincare`.
    - Actualizado `shaderVisualization.ts` con la definición del shader.

### 2. Documentación de Motores
- **Nueva Guía:** `docs/90_Troubleshooting/ENGINE_SWITCHING.md` explica cómo cambiar de motor usando comandos (`/switch_engine`) o argumentos de carga (`force_engine`).

### 3. Conceptos
- **Nuevo Concepto:** `docs/20_Concepts/The_Holographic_Viewer.md` explica la teoría detrás de la visualización.

## 🧠 Racional
La visualización de Poincaré es crucial para interpretar el grid 2D como el "borde" de un universo 3D emergente (Holographic Principle). El uso de shaders permite realizar esta transformación compleja en tiempo real sin impacto en el rendimiento del servidor.

## 🔗 Referencias
- [[The_Holographic_Viewer]]
- [[ENGINE_SWITCHING]]
