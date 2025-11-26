## 2025-11-20 - Checkpoint Step Tracking y Grid Scaling Info

### Contexto
Implementación de tracking del paso del checkpoint y información de escalado de grid para mostrar correctamente el paso inicial desde el checkpoint.

### Problemas Resueltos

#### 1. Paso Actual Siempre Empezaba en 0
- **Antes:** `simulation_step` siempre se inicializaba en 0, incluso si había un checkpoint con un paso guardado
- **Después:** Lee el paso del checkpoint y inicializa `simulation_step = checkpoint_step`

#### 2. Falta de Información del Grid en UI
- **Antes:** No se mostraba información sobre el escalado del grid (training vs inference)
- **Después:** Se muestra `training_grid_size` y `inference_grid_size` en `checkpoint_info`

#### 3. Visualización "Total - Actual"
- **Antes:** Solo se mostraba el paso total
- **Después:** Se muestra "total - relativo" con hover mostrando el paso del checkpoint

### Implementación

**Archivo:** `src/pipelines/pipeline_server.py`

**Cambios:**
- Lee `step` y `episode` del checkpoint antes de cargar el modelo
- Si no hay `step`, calcula desde `episode × steps_per_episode`
- Guarda `checkpoint_step`, `checkpoint_episode`, `initial_step` en `g_state`
- Incluye `checkpoint_info` en `inference_status_update` con información del grid

**Archivo:** `frontend/src/modules/Dashboard/components/Toolbar.tsx`

**Cambios:**
- Muestra "total - relativo" en lugar de solo el paso total
- Ejemplo: `"1,356 - 0"` (total 1356, relativo 0 desde checkpoint)
- Hover muestra información del checkpoint

### Archivos Modificados
1. **`src/pipelines/pipeline_server.py`** - Lectura y guardado de checkpoint info
2. **`frontend/src/modules/Dashboard/components/Toolbar.tsx`** - Visualización mejorada

### Estado
✅ **Completado**

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
