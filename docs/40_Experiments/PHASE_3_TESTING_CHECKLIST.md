# 🧪 Phase 3 Manual Testing Checklist

**Objetivo:** Verificar que el sistema de History Buffer y las Visualizaciones Avanzadas funcionan correctamente antes de merge a `main`.

**Branch:** `feat/phase-3-completion`  
**Fecha:** 2025-11-26

---

## 📋 Pre-requisitos

- [ ] Backend corriendo: `python src/cli.py dev --fast`
- [ ] Frontend corriendo: `cd frontend && npm run dev`
- [ ] Browser abierto en `http://localhost:5173`
- [ ] DevTools abiertos (Console + Network tab)

---

## 1. 🕰️ History Buffer System

### 1.1 Inicialización del Buffer
- [ ] **Cargar experimento** existente o crear uno nuevo
- [ ] **Iniciar simulación** (Play button)
- [ ] **Verificar** que los controles de History aparecen en el header
- [ ] **Verificar** que el contador de frames aumenta (0 → 1 → 2 → ...)

**Resultado Esperado:**
- Controles de History visibles
- Contador "Frame X/1000" actualizado en tiempo real
- Sin errores en Console

---

### 1.2 Navegación Temporal (Rewind)
- [ ] **Pausar simulación** (Pause button)
- [ ] **Slider de timeline**: Mover hacia atrás (hacia frame 0)
- [ ] **Verificar** que la visualización retrocede en el tiempo
- [ ] **Verificar** que el canvas muestra el estado del frame seleccionado
- [ ] **Verificar** que las métricas (Energy, Entropy, etc.) cambian según el frame

**Resultado Esperado:**
- Visualización cambia instantáneamente al mover el slider
- Estado cuántico restaurado correctamente
- Métricas consistentes con el frame seleccionado
- Sin lag perceptible (\< 100ms de latencia)

---

### 1.3 Restauración y Replay
- [ ] **Retroceder** al frame 50 (usando slider)
- [ ] **Click en "Restore & Resume"** button
- [ ] **Verificar** que la simulación se reanuda desde frame 50
- [ ] **Verificar** que el contador salta de 50 → 51 → 52...
- [ ] **Verificar** que la visualización es consistente (no salta/glitches)

**Resultado Esperado:**
- Simulación reanudada desde punto restaurado
- No hay discontinuidades visuales
- Estado cuántico coherente (no explosión de valores)

---

### 1.4 Buffer Completo (1000 frames)
- [ ] **Dejar simulación corriendo** hasta llenar buffer (frame 1000+)
- [ ] **Verificar** que frames antiguos se eliminan (frames 1-X desaparecen)
- [ ] **Verificar memory usage** en DevTools → Memory tab
  - Debería estar estable (~1-2GB dependiendo de grid size)
  - No debería crecer indefinidamente
- [ ] **Navegar por slider** desde frame 1000 hasta frame actual

**Resultado Esperado:**
- Buffer circular funciona (frames antiguos eliminados automáticamente)
- Memory usage estable
- Slider permite navegar por todo el rango disponible

---

### 1.5 Edge Cases
- [ ] **Restaurar mientras simulación corriendo** (sin pausar primero)
  - ¿Se pausa automáticamente o da error?
- [ ] **Mover slider muy rápido** (arrastrar de extremo a extremo varias veces)
  - ¿Se mantiene responsive?
  - ¿Causa lag o crash?
- [ ] **Cambiar grid size con buffer lleno**
  - ¿Se limpia el buffer correctamente?
  - ¿Hay memory leaks?

**Resultado Esperado:**
- No crashes
- Comportamiento predecible en todos los casos
- Mensajes de error claros si algo falla

---

## 2. 🎨 Advanced Field Visualizations

### 2.1 Backend Support Verification
- [ ] **Abrir selector de visualización** (dropdown "Visualization Type")
- [ ] **Verificar** que las opciones aparecen:
  - [ ] Densidad
  - [ ] **Parte Real** ✨
  - [ ] **Parte Imaginaria** ✨
  - [ ] Fase
  - [ ] **Fase HSV** ✨
  - [ ] Poincaré
  - [ ] Flow
  - [ ] Phase Attractor

**Resultado Esperado:**
- Todas las opciones visibles
- Nuevas opciones (Real, Imag, HSV) presentes

---

### 2.2 Parte Real Visualization
- [ ] **Seleccionar "Parte Real"** en dropdown
- [ ] **Verificar** que la visualización cambia
- [ ] **Esperar 2-3 segundos** (nueva visualización llega del backend)
- [ ] **Verificar colormap**: Blue (negativo) → Yellow (positivo)
- [ ] **Activar WebGL shader** (botón "Use Shader" en top-right si existe)
- [ ] **Verificar FPS** en DevTools Performance tab
  - WebGL: ~60 FPS esperado
  - Canvas2D: ~15-30 FPS esperado

**Resultado Esperado:**
- Visualización de Re(ψ) correcta
- Colormap blue-yellow adecuado
- WebGL shader activo (GPU rendering)
- FPS estable y alto

---

### 2.3 Parte Imaginaria Visualization
- [ ] **Seleccionar "Parte Imaginaria"** en dropdown
- [ ] **Verificar** que la visualización cambia
- [ ] **Verificar colormap**: Blue (negativo) → Yellow (positivo)
- [ ] **Comparar** con Parte Real (deberían ser diferentes pero complementarios)
- [ ] **Verificar FPS** con WebGL activo

**Resultado Esperado:**
- Visualización de Im(ψ) correcta
- Colormap consistente con Real
- GPU rendering activo
- FPS estable

---

### 2.4 Fase HSV Visualization (NEW - GPU Shader)
- [ ] **Seleccionar "Fase HSV"** en dropdown
- [ ] **Verificar** que aparece un color wheel:
  - Rojo (fase = 0°)
  - Amarillo (fase = 60°)
  - Verde (fase = 120°)
  - Cian (fase = 180°)
  - Azul (fase = 240°)
  - Magenta (fase = 300°)
- [ ] **Verificar** que los colores son vibrantes (saturation = 1.0, value = 1.0)
- [ ] **Verificar** que el shader WebGL está activo (botón "Use Shader")
- [ ] **Verificar FPS** con diferentes grid sizes:
  - 64x64: ~60 FPS
  - 256x256: ~60 FPS
  - 512x512: ~60 FPS (debería mantener buen rendimiento)

**Resultado Esperado:**
- Color wheel smooth y continuo (sin bandas/artefactos)
- GPU rendering activo (CRITICAL para performance)
- FPS alto y estable en todos los tamaños de grid
- 4-12x más rápido que Canvas2D fallback

---

### 2.5 Performance Comparison
- [ ] **Desactivar WebGL shader** (botón "Use Shader" OFF)
- [ ] **Cambiar a "Fase HSV"**
- [ ] **Medir FPS** en DevTools (Canvas2D fallback):
  - 256x256: ~15 FPS esperado
  - 512x512: ~5 FPS esperado
- [ ] **Activar WebGL shader** (botón "Use Shader" ON)
- [ ] **Medir FPS** con shader:
  - 256x256: ~60 FPS esperado
  - 512x512: ~60 FPS esperado
- [ ] **Calcular speedup**: FPS(WebGL) / FPS(Canvas2D)
  - Esperado: 4-12x speedup

**Resultado Esperado:**
- WebGL significativamente más rápido
- Performance gain evidente en grids grandes
- No visual artifacts con WebGL activo

---

### 2.6 Grid Size Stress Test
- [ ] **Crear simulación con grid 64x64**
- [ ] **Probar todas las visualizaciones** (density, real, imag, hsv)
- [ ] **Cambiar a grid 256x256**
- [ ] **Probar todas las visualizaciones**
- [ ] **Cambiar a grid 512x512** (si GPU lo soporta)
- [ ] **Probar todas las visualizaciones**
- [ ] **Verificar** que no hay degradación de performance
- [ ] **Verificar** que no hay memory leaks (ver DevTools → Memory)

**Resultado Esperado:**
- Todas las visualizaciones funcionan en todos los tamaños
- FPS se mantiene estable (WebGL shader)
- Memory usage crece proporcionalmente a grid size (esperado)
- No crashes ni OOM errors

---

## 3. 🔗 Integration Tests

### 3.1 History + Visualizations
- [ ] **Activar "Fase HSV"**
- [ ] **Correr simulación** por 100 frames
- [ ] **Pausar** y retroceder a frame 50
- [ ] **Verificar** que la visualización HSV se restaura correctamente
- [ ] **Cambiar a "Parte Real"** mientras estás en frame 50
- [ ] **Verificar** que el cambio funciona (backend calcula Real del frame 50)

**Resultado Esperado:**
- Cambio de visualización funciona en frames históricos
- Backend calcula correctamente la visualización del frame seleccionado
- No hay inconsistencias entre buffer y visualización

---

### 3.2 ROI + Advanced Vis
- [ ] **Activar "Fase HSV"**
- [ ] **Hacer zoom in** (zoom \> 1.1x)
- [ ] **Verificar** que ROI se activa automáticamente
- [ ] **Pan** por diferentes regiones del canvas
- [ ] **Verificar** que la visualización HSV se actualiza correctamente en cada región

**Resultado Esperado:**
- ROI funciona correctamente con visualizaciones avanzadas
- Performance se mantiene alta con ROI + WebGL
- Visualización correcta en todas las regiones

---

## 4. 🐛 Error Handling

### 4.1 Backend Disconnection
- [ ] **Detener backend** (Ctrl+C en terminal)
- [ ] **Verificar** que frontend muestra error de conexión
- [ ] **Reiniciar backend**
- [ ] **Verificar** que frontend se reconecta automáticamente
- [ ] **Verificar** que buffer se mantiene (o se limpia correctamente)

**Resultado Esperado:**
- Error handling claro
- Reconexión automática funciona
- No data corruption

---

### 4.2 Invalid Buffer State
- [ ] **Intentar restaurar frame** que no existe (ej: frame -1)
  - ¿Qué sucede?
- [ ] **Intentar navegar** antes de que haya buffer (frame 0 sin simulación corrida)
  - ¿Se deshabilitan los controles?

**Resultado Esperado:**
- Validación de inputs
- Mensajes de error claros
- UI no se rompe

---

## 5. ✅ Success Criteria

**Mínimo para Merge:**
- [ ] History Buffer funciona correctamente (1.1 - 1.4 completos)
- [ ] **Parte Real** renderiza correctamente (2.2 completo)
- [ ] **Parte Imaginaria** renderiza correctamente (2.3 completo)
- [ ] **Fase HSV** renderiza correctamente con GPU shader (2.4 completo)
- [ ] Performance gain con WebGL shader evidente (2.5 completo)
- [ ] No crashes ni memory leaks (2.6 completo)
- [ ] No blockers críticos (4.1 - 4.2 completos)

**Nice to Have:**
- [ ] Todos los edge cases manejados correctamente (1.5)
- [ ] ROI + Advanced Vis funciona sin problemas (3.2)
- [ ] FPS \> 50 en grid 512x512 con shader activo

---

## 📊 Testing Results Summary

**Tester:** _________________  
**Date:** 2025-11-26  
**Time Spent:** _______ minutes

**Overall Status:** ❓ PENDIENTE | ✅ PASSED | ❌ FAILED

**Critical Issues Found:** _________________

**Notes:**
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

---

**Next Steps:**
- [ ] Fix critical issues (if any)
- [ ] Update docs with findings
- [ ] Create PR for merge to `main`
- [ ] Request code review
