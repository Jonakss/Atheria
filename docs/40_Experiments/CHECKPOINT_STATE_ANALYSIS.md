---
title: Análisis de Checkpoints y Estados Guardados
type: analysis
status: active
tags: [checkpoints, state-management, persistence]
created: 2025-11-21
updated: 2025-11-21
related: [[30_Components/Training_Pipeline|Pipeline de Entrenamiento]], [[PENDING_TASKS|Tareas Pendientes]]
---

# Análisis de Checkpoints y Estados Guardados

**Fecha**: 2025-11-21  
**Objetivo**: Verificar qué se guarda en los checkpoints y si hay guardado de estados globales de simulación.

---

## 📊 Estado Actual

### ✅ Checkpoints de Entrenamiento

**Ubicación**: `output/training_checkpoints/<experiment_name>/checkpoint_ep<X>.pth`

**Contenido**:
```python
{
    'episode': int,                    # Episodio actual
    'model_state_dict': dict,          # Pesos del modelo
    'optimizer_state_dict': dict,      # Estado del optimizador
    'scheduler_state_dict': dict,      # Estado del scheduler
    'loss': float,                     # Pérdida total
    'metrics': dict,                   # Métricas (survival, symmetry, complexity)
    'combined_metric': float           # Métrica combinada para ordenamiento
}
```

**Lo que NO incluye**:
- ❌ `step` o `simulation_step` actual (se calcula desde `episode * steps_per_episode`)
- ❌ Estado de simulación (`psi` actual)
- ❌ Estado global (`g_state`)
- ❌ Configuración de visualización
- ❌ Estado del motor (excepto modelo)

**Guardado**: Automático cada `SAVE_EVERY_EPISODES` (típicamente cada 10 episodios)

---

### ❌ Estados de Simulación

**Estado Actual**: NO se guardan automáticamente

**Métodos Disponibles**:
1. **`motor.save_state(filepath)`** (solo `Aetheria_Motor`):
   - Guarda: `psi`, `h_state`, `c_state` (si existe)
   - NO se llama automáticamente
   - Solo guarda estado del motor, no `g_state` ni configuración

2. **Snapshots Manuales** (`handle_capture_snapshot`):
   - Guarda: `psi`, `step`, `timestamp`
   - Solo en memoria: `g_state['snapshots']`
   - NO se persisten a disco automáticamente
   - Límite: 500 snapshots en memoria

3. **Historial** (`SimulationHistory`):
   - Guarda: `step`, `timestamp`, `map_data`, `hist_data`
   - Se puede guardar manualmente a archivo JSON
   - NO guarda `psi` completo (solo `map_data` procesado)
   - Ubicación: `output/simulation_history/`

---

## 🔍 Problemas Identificados

### 1. No se Guarda Estado de Simulación en Checkpoints
**Problema**:
- Los checkpoints solo guardan el modelo entrenado
- Al cargar un checkpoint, la simulación siempre empieza desde `step=0` o calculado desde `episode`
- No se puede "resumir" una simulación desde un punto específico

**Impacto**:
- Si se cierra el servidor durante una simulación larga, se pierde el progreso
- No se puede continuar una simulación desde un step específico
- Los checkpoints no son "snapshots completos" de la simulación

### 2. Snapshots Solo en Memoria
**Problema**:
- Los snapshots se guardan en `g_state['snapshots']`
- Si el servidor se cierra, se pierden
- No hay persistencia automática

**Impacto**:
- No se pueden revisar snapshots de sesiones anteriores
- No se puede analizar evolución temporal de simulaciones pasadas

### 3. No se Guarda Estado Global (`g_state`)
**Problema**:
- `g_state` contiene configuración importante:
  - `viz_type`, `simulation_step`, `is_paused`
  - `simulation_speed`, `target_fps`, `frame_skip`
  - `live_feed_enabled`, `data_compression_enabled`
  - Configuración de ROI, análisis, etc.
- No se persiste a disco

**Impacto**:
- Cada vez que se reinicia el servidor, se pierde la configuración
- No se puede "resumir" exactamente donde se dejó

---

## 💡 Soluciones Propuestas

### Opción 1: Guardar Estado de Simulación en Checkpoints (Recomendado)
**Implementación**:
- Agregar `simulation_state` al checkpoint de entrenamiento:
  ```python
  checkpoint_data = {
      # ... datos existentes ...
      'simulation_step': int,          # Step actual
      'psi': torch.Tensor,             # Estado cuántico actual
      'motor_state': dict,             # Estado del motor (si aplica)
  }
  ```

**Ventajas**:
- Checkpoints completos (modelo + estado)
- Se puede resumir simulación exactamente donde se dejó
- Un solo archivo contiene todo

**Desventajas**:
- Checkpoints más grandes (psi puede ser ~100MB para grid 256x256)
- Solo funciona si se guarda durante simulación (no solo entrenamiento)

### Opción 2: Guardar Estados de Simulación Separados
**Implementación**:
- Nuevo sistema de "snapshots persistentes":
  - Guardar `psi`, `step`, configuración en archivos separados
  - Ubicación: `output/simulation_snapshots/<experiment_name>/step_<X>.pt`
  - Permitir guardar automáticamente cada N steps

**Ventajas**:
- Separación de concerns (entrenamiento vs simulación)
- Se pueden guardar múltiples snapshots sin afectar checkpoints
- Más flexible para análisis

**Desventajas**:
- Más archivos para gestionar
- Requiere sistema de limpieza de snapshots antiguos

### Opción 3: Guardar Estado Global Configuración
**Implementación**:
- Guardar `g_state` relevante en archivo JSON:
  - `output/simulation_states/<experiment_name>/last_state.json`
  - Solo configuraciones (no tensores)
  - Guardar automáticamente al pausar o cada N steps

**Ventajas**:
- Ligero (solo texto JSON)
- Permite resumir configuración
- Compatible con versionado

**Desventajas**:
- No incluye `psi` (solo configuración)
- No es un snapshot completo

---

## 📋 Recomendación

**Implementar Opción 1 + Opción 2 (Híbrido)**:

1. **Guardar `simulation_step` en checkpoints** (solo número, ligero)
   - Ya casi está (se calcula desde episode, pero debería guardarse explícitamente)

2. **Sistema de Snapshots Persistentes** (Opción 2)
   - Guardar `psi` completo en archivos separados
   - Permitir guardado automático cada N steps (configurable)
   - Integrar con UI para guardar/cargar snapshots

3. **Guardar Configuración de Simulación** (Opción 3)
   - Guardar `g_state` relevante en JSON
   - Al cargar checkpoint, restaurar configuración si existe

---

## 🔗 Referencias

- `src/trainers/qc_trainer_v4.py:222` - Función `save_checkpoint()`
- `src/engines/qca_engine.py:74` - Función `save_state()` del motor
- `src/pipelines/pipeline_server.py:3387` - Handler `handle_capture_snapshot()`
- `src/managers/history_manager.py` - Sistema de historial

---

**Última actualización**: 2025-11-21

