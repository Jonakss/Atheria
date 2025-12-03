# 📋 Tareas Pendientes - Atheria 4

**Última actualización:** 2024-11-21  
**Estado General:** 🟢 **Críticos resueltos, solo funcionalidades opcionales pendientes**

---

## 🔴 CRÍTICO - Problemas del Motor Nativo

### 1. Motor Nativo se Cuelga/Bloquea
**Prioridad:** 🔴 **CRÍTICA**  
**Estado:** ✅ **RESUELTO** (2024-12-20)

**Problema:**
- El motor nativo se quedaba bloqueado durante la simulación
- No respondía a comandos de pausa inmediatamente
- Requería matar el proceso para detener

**Solución Implementada:**
- ✅ Lazy conversion implementada: solo convierte cuando se necesita visualizar
- ✅ Verificación de pausa durante conversión disperso→denso
- ✅ Check de pausa dentro del loop de simulación en Python

**Referencias:**
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES#Motor Nativo se Cuelga/Bloquea]]
- [[AI_DEV_LOG#Optimizaciones Críticas Motor Nativo Implementadas]]

---

- [x] **[CRITICAL]** Debug Native Engine Freeze (Warmup)
    - **Status**: ✅ **FIXED** (2025-12-03)
    - **Cause**: Deadlock due to `torch::set_num_threads(1)` inside OpenMP region.
    - **Resolution**: Removed problematic call.
- [ ] **[CRITICAL]** Optimize Native Engine Performance
    - **Status**: 🔴 **BLOCKER**
    - **Context**: Native engine is >100x slower than Python on CPU for dense grids.
    - **Action**: Profile and optimize `step_native` loop, specifically batch construction and map access. Consider alternative data structures (e.g., dense blocks) for high-density regions.

### 2. Lentitud Extrema en Tiempo Real
**Prioridad:** 🔴 **CRÍTICA**  
**Estado:** ✅ **RESUELTO** (2024-12-20)

**Problema:**
- El motor nativo se ponía muy lento en tiempo real
- FPS caía dramáticamente
- UI se congelaba

**Solución Implementada:**
- ✅ Lazy Conversion: Solo convierte cuando se necesita visualizar (`get_dense_state()`)
- ✅ ROI para Conversión: Solo convierte región visible (reducción de 65,536 a ~10,000-20,000 coordenadas)
- ✅ Pause check durante conversión: Permite pausa inmediata incluso durante conversión larga
- ⏳ Batch Conversion en C++: Pendiente (opcional, mejora adicional)

**Resultados:**
- Motor nativo ahora alcanza ~10,000 steps/segundo
- Conversión solo se ejecuta cuando se necesita (lazy)
- ROI reduce overhead de conversión en 3-5x

**Referencias:**
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES#Lentitud Extrema en Tiempo Real]]
- [[AI_DEV_LOG#Optimizaciones Críticas Motor Nativo Implementadas]]

---

## 🟡 ALTO - Funcionalidades Faltantes

### 3. Mostrar "Paso Actual" como "Total - Actual"
**Prioridad:** 🟡 **ALTA**  
**Estado:** ✅ **RESUELTO** (2024-12-20)

**Requisito:**
- Mostrar "total - actual" desde que se continuó
- Hover mostrando punto de inicio: "Se inició desde paso X"

**Implementación:**
- ✅ Display actualizado en `Toolbar.tsx`: Muestra "total - relativo" cuando hay `initial_step`
- ✅ Hover muestra información del checkpoint (episodio y paso)
- ✅ Backend envía `initial_step`, `checkpoint_step`, `checkpoint_episode` en `simulation_info`

**Ubicación:**
- `frontend/src/modules/Dashboard/components/Toolbar.tsx` (líneas 115-145)
- `src/pipelines/pipeline_server.py` - Envía punto de inicio en `simulation_info`

---

### 4. Visualizaciones en Shaders (GPU)
**Prioridad:** 🟡 **ALTA**  
**Estado:** ⏳ **EN VERIFICACIÓN** (2025-11-22)

**Requisito:**
- Cuando GPU está disponible, usar shaders para visualizaciones
- Evitar cuellos de botella en CPU
- Liberar CPU para simulación

**Implementación:**
- Usar Three.js shaders o WebGL para procesamiento
- Procesar visualización directamente en GPU
- Solo transferir datos necesarios a CPU

---

### 5. Apagar Servidor desde UI
**Prioridad:** 🟡 **ALTA**  
**Estado:** ✅ **IMPLEMENTADO** (2024-11-20)

**Requisito:**
- Botón en UI para apagar el servidor
- Confirmación antes de apagar
- Guardar estado antes de apagar (opcional)

**Implementación:**
- ✅ Handler `handle_shutdown()` creado en backend
- ✅ Comando WebSocket: `server.shutdown` agregado a HANDLERS
- ✅ Botón "Apagar Servidor" en SettingsPanel (con confirmación)
- ✅ shutdown_event expuesto en g_state para acceso desde handlers

**Ubicación:**
- Backend: `src/pipelines/pipeline_server.py` - `handle_shutdown()` (líneas ~2147-2178)
- Frontend: `frontend/src/modules/Dashboard/components/SettingsPanel.tsx` - Sección "Control del Servidor"

---

### 6. Migración Automática de Estado al Cambiar de Engine
**Prioridad:** 🟡 **ALTA**  
**Estado:** ✅ **IMPLEMENTADO** (2024-12-20)

**Requisito:**
- Cuando se cambia de engine y está pausado, migrar estado automáticamente
- Preservar `current_step` y `psi` si es posible

**Implementación:**
- ✅ `handle_switch_engine()` implementado en `pipeline_server.py`
- ✅ Preserva `current_step` y `psi` al cambiar de engine
- ✅ Pausa y reanuda simulación automáticamente durante el cambio
- ✅ Limpieza explícita de motor anterior para evitar segfaults

**Ubicación:**
- `src/pipelines/pipeline_server.py` - `handle_switch_engine()` (líneas ~1845-1950)

---

## 🟢 MEDIO - Mejoras y Optimizaciones

### 7. Exportación Automática de Modelos a TorchScript
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Parcialmente implementado

**Requisito:**
- Exportar automáticamente al cargar experimento si no existe modelo JIT
- Ya implementado, pero puede mejorarse:
  - Mejor manejo de errores
  - Progress indicator en UI
  - Cache de modelos exportados

---

### 8. Snapshots Durante Entrenamiento
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Pendiente

**Requisito:**
- Capturar snapshots automáticamente durante entrenamiento
- Guardar en directorio de checkpoints
- Permitir revisar snapshots en UI

---

### 9. Sistema de Historial/Buffer Completo
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Parcialmente implementado

**Requisito:**
- Navegación temporal (rewind/replay)
- Buffer completo de estados
- Navegación con teclado/UI

**Estado Actual:**
- `simulation_history` existe pero no está completamente integrado

---

### 10. Más Visualizaciones de Campos
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Pendiente

**Requisito:**
- Real/Imaginario separados
- Fase HSV avanzada
- Más opciones de visualización

---

## 📚 DOCUMENTACIÓN PENDIENTE

### 11. Documentar Fase 4
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Pendiente

**Requisito:**
- Documentar arquitectura de Fase 4
- Actualizar roadmap
- Documentar nuevas funcionalidades

---

### 12. Conectar EpochDetector Completamente
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Parcialmente implementado

**Requisito:**
- Conectar EpochDetector al dashboard completamente
- Visualizaciones de épocas
- Transiciones de época en UI

---

## 🔧 OPTIMIZACIONES TÉCNICAS

### 13. Integrar Quadtree/Octree en Motor
**Prioridad:** 🟢 **BAJA** (Opcional)  
**Estado:** Pendiente

**Requisito:**
- Integrar índice espacial en motor de simulación
- Mejorar búsqueda de vecinos
- Optimizar para simulaciones grandes

---

### 14. Benchmark Completo Python vs C++
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Pendiente

**Requisito:**
- Comparar rendimiento Python vs C++
- Medir tiempo de `step()` para diferentes tamaños
- Medir uso de memoria
- Documentar resultados

---

### 15. Paralelismo (OpenMP/std::thread)
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Pendiente

**Requisito:**
- Paralelizar `step_native()` en C++
- Usar OpenMP o std::thread
- Mejorar rendimiento para simulaciones grandes

---

### 16. Memory Pools
**Prioridad:** 🟢 **MEDIA**  
**Estado:** Pendiente

**Requisito:**
- Implementar memory pools en C++
- Reducir allocaciones/deallocations
- Mejorar rendimiento

---

## 🟣 EXPERIMENTAL - Nuevos Conceptos

### 17. Exploración del Principio Holográfico
**Prioridad:** 🟣 **EXPERIMENTAL**
**Estado:** 💡 **CONCEPTO**

**Requisito:**
- Investigar implementación de `HolographicEngine` (Física 2D -> 3D)
- Prototipar `HolographicViewer` para visualizar la frontera y el bulk
- Experimentar con "Ley M" en el borde del caos usando codificación holográfica

**Referencias:**
- [[HOLOGRAPHIC_PRINCIPLE]]

---

## 📊 RESUMEN POR PRIORIDAD

### 🔴 CRÍTICO (Implementar Inmediatamente)
~~1. Motor Nativo se Cuelga/Bloquea~~ ✅ **RESUELTO**
~~2. Lentitud Extrema en Tiempo Real~~ ✅ **RESUELTO**

### 🟡 ALTO (Implementar Pronto)
~~3. Mostrar "Paso Actual" como "Total - Actual"~~ ✅ **RESUELTO**
4. Visualizaciones en Shaders (GPU) - ⏳ **EN ROADMAP** (Phase 2 - Opcional)
~~5. Apagar Servidor desde UI~~ ✅ **IMPLEMENTADO** (2024-11-20)
~~6. Migración Automática de Estado~~ ✅ **IMPLEMENTADO**
~~9. Sistema de Historial/Buffer Completo~~ ✅ **IMPLEMENTADO** (2024-11-21)

### 🟢 MEDIO/BAJO (Implementar Después)
7-16. Resto de tareas

---

## 🔗 Referencias

- [[NATIVE_ENGINE_PERFORMANCE_ISSUES]] - Problemas de rendimiento del motor nativo
- [[AI_DEV_LOG]] - Log de desarrollo
- [[Native_Engine_Core]] - Documentación del motor nativo
- [[ROADMAP_PHASE_1]] - Roadmap de Fase 1

---

**Nota:** Este documento se actualiza regularmente. Última actualización: 2024-12-20

