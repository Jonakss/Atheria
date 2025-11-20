# 📋 Tareas Pendientes - Atheria 4

**Última actualización:** 2024-12-20  
**Estado General:** 🔴 **Varios problemas críticos pendientes**

---

## 🔴 CRÍTICO - Problemas del Motor Nativo

### 1. Motor Nativo se Cuelga/Bloquea
**Prioridad:** 🔴 **CRÍTICA**  
**Estado:** Pendiente

**Problema:**
- El motor nativo se queda bloqueado durante la simulación
- No responde a comandos de pausa inmediatamente
- Requiere matar el proceso para detener

**Causa:**
- `step_native()` en C++ es bloqueante y no verifica pausa
- `_update_dense_state_from_sparse()` se ejecuta en cada paso y puede tomar mucho tiempo
- No hay verificación de pausa durante la ejecución

**Solución Propuesta:**
- Implementar lazy conversion (solo convertir cuando se necesita)
- Agregar verificación de pausa durante conversión disperso→denso
- Verificar pausa dentro de `step_native()` en C++ (si es posible)

**Referencias:**
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES#Motor Nativo se Cuelga/Bloquea]]

---

### 2. Lentitud Extrema en Tiempo Real
**Prioridad:** 🔴 **CRÍTICA**  
**Estado:** Pendiente

**Problema:**
- El motor nativo se pone muy lento en tiempo real
- FPS cae dramáticamente
- UI se congela

**Causa:**
- Conversión completa en cada paso: itera sobre **todo el grid** (256x256 = 65,536 coordenadas)
- 65,536 llamadas a `get_state_at()` en cada paso
- Overhead Python↔C++ × 65,536 = MUY COSTOSO

**Solución Propuesta:**
1. **Lazy Conversion** (Prioridad Alta): Solo convertir cuando se necesita visualizar
2. **ROI para Conversión** (Prioridad Alta): Solo convertir región visible
3. **Batch Conversion en C++** (Prioridad Media): Reducir overhead Python↔C++
4. **Cache de Estado Denso** (Prioridad Baja): Reutilizar conversión si estado no cambió

**Referencias:**
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES#Lentitud Extrema en Tiempo Real]]

---

## 🟡 ALTO - Funcionalidades Faltantes

### 3. Mostrar "Paso Actual" como "Total - Actual"
**Prioridad:** 🟡 **ALTA**  
**Estado:** Pendiente

**Requisito:**
- Mostrar "total - actual" desde que se continuó
- Hover mostrando punto de inicio: "Se inició desde paso X"

**Ubicación:**
- `frontend/src/modules/Dashboard/components/Toolbar.tsx` - Actualizar display de paso
- `src/pipelines/pipeline_server.py` - Guardar punto de inicio al cargar experimento

---

### 4. Visualizaciones en Shaders (GPU)
**Prioridad:** 🟡 **ALTA**  
**Estado:** Pendiente

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
**Estado:** Pendiente

**Requisito:**
- Botón en UI para apagar el servidor
- Confirmación antes de apagar
- Guardar estado antes de apagar (opcional)

**Implementación:**
- Nuevo comando WebSocket: `server.shutdown`
- Handler en backend que llama a `asyncio.get_event_loop().stop()`
- Botón en UI (SettingsPanel o similar)

---

### 6. Migración Automática de Estado al Cambiar de Engine
**Prioridad:** 🟡 **ALTA**  
**Estado:** Parcialmente implementado

**Requisito:**
- Cuando se cambia de engine y está pausado, migrar estado automáticamente
- Preservar `current_step` y `psi` si es posible

**Estado Actual:**
- Ya implementado en `handle_switch_engine()` - líneas 1552-1577
- **Pendiente:** Verificar que funciona correctamente con diferentes tamaños de grid

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

## 📊 RESUMEN POR PRIORIDAD

### 🔴 CRÍTICO (Implementar Inmediatamente)
1. Motor Nativo se Cuelga/Bloquea
2. Lentitud Extrema en Tiempo Real

### 🟡 ALTO (Implementar Pronto)
3. Mostrar "Paso Actual" como "Total - Actual"
4. Visualizaciones en Shaders (GPU)
5. Apagar Servidor desde UI
6. Migración Automática de Estado (verificar)

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

