## 2025-11-20 - Modo Manual de Visualización (steps_interval = 0)

### Contexto
El usuario necesita una forma de dejar la simulación corriendo sin enviar frames automáticamente, solo actualizar la visualización cuando se presione un botón manualmente. Esto permite:
- Ejecutar la simulación a máxima velocidad sin overhead de visualización
- Reducir el uso de ancho de banda y recursos
- Mantener el control manual sobre cuándo actualizar la visualización

### Problema Resuelto

#### Antes
- `steps_interval` tenía un mínimo de 1, siempre enviaba frames automáticamente
- No había forma de ejecutar la simulación sin enviar frames periódicamente

#### Después
- `steps_interval = 0` activa el **modo manual**: la simulación corre sin enviar frames
- Nuevo handler `handle_update_visualization()` para actualización manual con botón
- El estado se mantiene en `g_state` para que al reconectar se vea el progreso

### Implementación

#### 1. Modo Manual (`steps_interval = 0`)

**Archivo:** `src/pipelines/pipeline_server.py`

**Función:** `handle_set_steps_interval()`

**Cambios:**
- Ahora acepta `steps_interval = 0` (anteriormente mínimo era 1)
- Límite máximo aumentado de `1,000` a `1,000,000` (1 millón)
- `steps_interval = 0` → Modo manual: no enviar frames automáticamente
- `steps_interval > 0` → Modo automático: enviar frame cada N pasos (hasta 1 millón)

**Lógica en `simulation_loop()`:**
```python
if steps_interval == 0:
    # Modo manual: ejecutar pasos rápidamente sin enviar frames
    steps_to_execute = 100  # Ejecutar múltiples pasos para velocidad
    should_send_frame = (g_state['last_frame_sent_step'] == -1)  # Solo primer frame
else:
    # Modo automático: enviar frame cada N pasos
    steps_to_execute = steps_interval
    should_send_frame = (steps_interval_counter >= steps_interval)
```

**Límite de `steps_interval`:**
- **Mínimo:** `0` (modo manual)
- **Máximo:** `1,000,000` (1 millón de pasos)
- Permite configurar intervalos muy grandes (ej: cada 10,000 o 100,000 pasos) para ejecuciones largas

#### 2. Handler de Actualización Manual

**Función:** `handle_update_visualization()`

**Características:**
- Actualiza la visualización manualmente cuando se presiona el botón
- Soporta motor nativo (lazy conversion) y motor Python
- Aplica optimizaciones (ROI, compresión, downsampling)
- Envía frame a todos los clientes conectados

**Uso:**
```javascript
// Frontend puede llamar:
ws.send(JSON.stringify({
    scope: "simulation",
    command: "update_visualization",
    args: {}
}));
```

**Mensajes de log:**
- Modo manual: `"[Simulación] Paso X completado (modo manual: presiona 'Actualizar Visualización' para ver)"`
- Modo automático: `"[Simulación] Paso X completado (live feed desactivado, mostrando cada N pasos)"`

### Estado Persistente

#### ¿Se mantiene el progreso al desconectar?

✅ **Sí**, el estado se mantiene en `g_state`:
- `g_state['simulation_step']` → Paso actual de la simulación
- `g_state['initial_step']` → Paso inicial (checkpoint)
- `g_state['motor']` → Motor con el estado cuántico actual
- `g_state['active_experiment']` → Experimento activo

**Al reconectar:**
- El cliente recibe `initial_state` con el estado actual
- El `step` se sincroniza automáticamente
- La visualización puede actualizarse manualmente para ver el estado actual

### Consideraciones de Seguridad

⚠️ **IMPORTANTE:** Ejecutar la simulación en modo manual sin supervisión puede ser **peligroso**:

1. **Uso de Recursos:**
   - La simulación consume CPU/GPU continuamente
   - Puede generar calor excesivo en el hardware
   - Puede afectar el rendimiento del sistema

2. **Memoria:**
   - Si la simulación se ejecuta por mucho tiempo, puede acumular memoria
   - Los snapshots automáticos pueden llenar el disco si están habilitados

3. **Recomendaciones:**
   - ✅ Usar `handle_enable_snapshots()` para controlar capturas automáticas
   - ✅ Monitorear el uso de recursos (CPU, GPU, RAM)
   - ✅ Establecer límites de tiempo de ejecución si es necesario
   - ✅ Pausar la simulación cuando no se esté usando

4. **Mejoras Futuras:**
   - [ ] Límite de tiempo de ejecución automático
   - [ ] Guardado automático periódico del estado
   - [ ] Monitoreo de recursos y alertas

### Archivos Modificados
1. **`src/pipelines/pipeline_server.py`**:
   - `handle_set_steps_interval()` - Ahora acepta `steps_interval = 0`
   - `handle_update_visualization()` - Nuevo handler para actualización manual
   - `simulation_loop()` - Lógica para modo manual
   - `HANDLERS` - Agregado `"update_visualization"` a `simulation`

### Corrección de Error

**Problema:** `UnboundLocalError: local variable 'logging' referenced before assignment` en `pipeline_viz.py`

**Causa:** Múltiples `import logging` dentro de bloques `except` hacían que Python tratara `logging` como variable local en toda la función.

**Solución:** Eliminados todos los `import logging` locales innecesarios (líneas 271, 296, 318, 535). `logging` ya está importado al inicio del archivo.

### Referencias
- `src/pipelines/pipeline_server.py` - Líneas 2707-2737 (handle_set_steps_interval)
- `src/pipelines/pipeline_server.py` - Líneas 983-1060 (handle_update_visualization)
- `src/pipelines/pipeline_server.py` - Líneas 221-325 (simulation_loop - modo manual)
- `src/pipelines/pipeline_viz.py` - Corrección de imports de logging

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
