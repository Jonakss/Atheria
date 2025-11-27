# 2025-11-26 - Feature: History Buffer System (Rewind/Replay)

## Contexto
Implementación completa del sistema de buffer circular en memoria para navegación temporal (rewind/replay) de simulaciones cuánticas. Permite retroceder a cualquier punto de los últimos 1000 frames sin re-ejecutar la simulación.

## Motivación
- **Debugging eficiente**: Inspeccionar comportamiento de simulación en puntos específicos
- **Exploración temporal**: Navegar libremente por la historia de la simulación
- **Análisis científico**: Comparar estados en diferentes momentos sin pérdida de datos

## Arquitectura Implementada

### 1. Backend - Buffer Circular Eficiente ✅

**Archivo:** `src/managers/history_manager.py`

**Cambios principales:**
- Refactorizado `SimulationHistory` para usar `collections.deque(maxlen=1000)`
- Operaciones O(1) para append/pop (vs O(n) con listas Python)  
- Almacenamiento de estado cuántico completo (`psi`) en CPU
- Auto-eliminación de frames antiguos al superar límite

**Decisión clave - `psi` en CPU:**
- ✅ Evita saturar VRAM del GPU
- ✅ Permite buffer más grande (1000 frames vs ~100 en GPU)
- ✅ Transferencia rápida GPU→CPU→GPU solo al restaurar

**Código relevante:**
```python
def add_frame(self, frame_data: Dict):
    # Detach psi to CPU to avoid VRAM saturation
    if 'psi' in frame_data and frame_data['psi'] is not None:
        import torch
        if isinstance(frame_data['psi'], torch.Tensor):
            frame_data['psi'] = frame_data['psi'].detach().cpu()
    
    self.frames.append(frame_data)  # O(1) with deque
```

### 2. Integración con Simulation Loop ✅

**Archivo:** `src/pipelines/core/simulation_loop.py`

**Mecanismo de Rewind:**
1.  Frontend envía `rewind_to_step(target_step)`
2.  Backend busca frame más cercano en buffer
3.  **Restauración de Estado:**
    - Carga `psi` del buffer
    - Mueve `psi` de CPU a GPU (`.to(device)`)
    - Actualiza `engine.state` y `g_state.step_count`
4.  Simulación continúa desde ese punto exacto

### 3. Frontend - UI de Control Temporal ✅

**Archivo:** `frontend/src/modules/SimulationControls/SimulationControls.tsx`

**Componentes:**
- **Slider de Historia:** Permite arrastrar para viajar en el tiempo
- **Botones de Playback:** Pause, Play, Step Forward, Rewind
- **Indicador de Buffer:** Muestra cuántos frames hay disponibles para rewind

## Pruebas Realizadas
1.  **Test de Memoria:** Ejecución continua por 1 hora. Uso de RAM estable (gracias a `deque` y limpieza automática).
2.  **Test de VRAM:** Rewind repetido no incrementa uso de VRAM (gracias a `psi` en CPU).
3.  **Test de Precisión:** Al hacer rewind y volver a simular, los resultados son deterministas y idénticos.

## Siguientes Pasos
- [ ] Implementar persistencia en disco para sesiones largas (>1000 frames)
- [ ] Añadir compresión de estados para optimizar RAM
