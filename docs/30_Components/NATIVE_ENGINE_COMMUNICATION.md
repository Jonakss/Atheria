# Comunicación Motor Nativo (C++) ↔ Python ↔ Frontend

## Arquitectura de Comunicación

```
┌─────────────────┐
│  C++ (LibTorch) │
│ atheria_core    │
│  .Engine        │
└────────┬────────┘
         │ PyBind11
         │ (tensors directos)
         ▼
┌─────────────────┐
│   Python        │
│NativeEngineWrap │
│     per         │
└────────┬────────┘
         │
         │ g_state['motor']
         │ motor.evolve_internal_state()
         ▼
┌─────────────────┐
│ Python Backend  │
│pipeline_server  │
│simulation_loop  │
└────────┬────────┘
         │ WebSocket (JSON)
         │ inference_status_update
         │ compile_status.is_native
         ▼
┌─────────────────┐
│   Frontend      │
│ React/TypeScript│
│  WebSocketCtx   │
└─────────────────┘
```

## Flujo de Datos

### 1. Carga del Motor (`handle_load_experiment`)

**Ubicación**: `src/pipeline_server.py` (línea ~720)

**Proceso**:

1. **Verificación de Configuración**:
   ```python
   use_native_engine = getattr(global_cfg, 'USE_NATIVE_ENGINE', True)
   ```

2. **Intento de Carga Nativo**:
   - Verifica si `atheria_core` está disponible
   - Busca modelo JIT (`.pt`) o exporta automáticamente desde checkpoint
   - Instancia `NativeEngineWrapper` si el modelo JIT existe
   - Carga el modelo en el motor C++: `motor.native_engine.load_model(jit_path)`

3. **Fallback a Python**:
   - Si falla la carga nativa, usa `Aetheria_Motor` (Python)
   - Compila con `torch.compile()` si está disponible

4. **Notificación al Frontend**:
   ```python
   compile_status = {
       "is_compiled": True,  # Nativo siempre "compilado"
       "is_native": True,    # ← INDICADOR PRINCIPAL
       "model_name": "Native Engine (C++)",
       "compiles_enabled": True
   }
   
   await broadcast({
       "type": "inference_status_update",
       "payload": {
           "status": "paused",
           "model_loaded": True,
           "experiment_name": exp_name,
           "compile_status": compile_status  # ← Información del motor
       }
   })
   ```

### 2. Evolución del Estado (`simulation_loop`)

**Ubicación**: `src/pipeline_server.py` (línea ~127)

**Proceso para Motor Nativo**:

```python
# Línea ~213
g_state['motor'].evolve_internal_state()
```

**Motor Nativo** (`NativeEngineWrapper.evolve_internal_state`):
```python
# src/engines/native_engine_wrapper.py (línea ~97)
def evolve_internal_state(self):
    # 1. Ejecutar paso nativo en C++ (TODO en C++)
    particle_count = self.native_engine.step_native()
    
    # 2. Convertir estado disperso → denso para visualización
    self._update_dense_state_from_sparse()
```

**Flujo Detallado del Motor Nativo**:

1. **C++ (`step_native`)**:
   - Itera sobre `SparseMap` de partículas
   - Genera vacío cuántico con `HarmonicVacuum` para vecinos
   - Batchea inputs para el modelo
   - Ejecuta inferencia: `model.forward({input})`
   - Actualiza partículas en el mapa disperso
   - **TODO ocurre en C++/GPU sin pasar por Python**

2. **Python (Conversión para Visualización)**:
   ```python
   # _update_dense_state_from_sparse (línea ~111)
   # Itera sobre todo el grid (256x256 por defecto)
   for y in range(grid_size):
       for x in range(grid_size):
           coord = atheria_core.Coord3D(x, y, 0)
           state_tensor = self.native_engine.get_state_at(coord)
           # Copiar a grid denso para frontend
           self.state.psi[0, y, x] = state_tensor
   ```
   
   **NOTA**: Esta conversión es el único cuello de botella. El motor nativo ejecuta la física 250-400x más rápido, pero la conversión disperso→denso toma tiempo.

3. **Visualización**:
   - `simulation_loop` obtiene `motor.state.psi` (denso)
   - Calcula visualizaciones con `get_visualization_data()`
   - Envía frame JSON al frontend via WebSocket

**Motor Python** (`Aetheria_Motor.evolve_internal_state`):
```python
# Todo en Python/PyTorch
# Más lento pero sin conversión de formato
```

### 3. Recepción en Frontend

**Ubicación**: `frontend/src/context/WebSocketContext.tsx`

**Procesamiento**:

```typescript
// Manejo de mensaje inference_status_update
case 'inference_status_update':
    const payload = message.payload;
    setInferenceStatus(payload.status);
    
    // compile_status contiene información del motor
    if (payload.compile_status) {
        const { is_native, model_name, is_compiled } = payload.compile_status;
        // ⚠️ ACTUALMENTE NO SE MUESTRA EN LA UI
        // Pero está disponible en el contexto
    }
    break;
```

## Cómo Verificar qué Motor Está en Uso

### 1. **Desde el Backend (Logs)**:

Cuando cargas un experimento, busca en los logs:

```
✅ Motor nativo (C++) cargado exitosamente con modelo JIT
⚡ Motor nativo cargado (250-400x más rápido)
```

O:

```
Usando motor Python tradicional (Aetheria_Motor)
✅ Modelo compilado con torch.compile()
```

### 2. **Desde la Configuración**:

```python
# src/config.py (línea ~74)
USE_NATIVE_ENGINE = True  # Por defecto True
```

Si es `True` y hay modelo JIT disponible, se usará el motor nativo.

### 3. **Desde el Frontend (Actualmente No Visible)**:

El frontend recibe `compile_status.is_native` pero **NO se muestra en la UI actualmente**.

## Problemas Identificados

### 1. **Falta Indicador Visual en Frontend**

El frontend recibe `compile_status.is_native` pero no lo muestra. Deberíamos:

- Agregar un badge en `MainHeader` o `ExperimentInfo` que muestre "⚡ Nativo" o "🐍 Python"
- Mostrar esto en `ExperimentInfo.tsx` junto con otros detalles del modelo

### 2. **Conversión Disperso → Denso Costosa**

El `_update_dense_state_from_sparse()` itera sobre **todo el grid** (256x256 = 65,536 coordenadas) en cada paso. Esto puede ser más lento que la simulación misma.

**Optimización Futura**:
- Solo convertir coordenadas activas
- Usar batching más agresivo
- Paralelizar la conversión con multiprocessing

### 3. **Falta Información de Rendimiento**

No hay forma de ver en tiempo real:
- Cuánto tiempo toma `step_native()` en C++
- Cuánto tiempo toma la conversión disperso→denso
- FPS real de la simulación

## Mejoras Propuestas

1. **Agregar Indicador Visual**:
   ```typescript
   // ExperimentInfo.tsx
   {compileStatus?.is_native && (
       <Badge color="blue" variant="light">
           ⚡ Motor Nativo (C++)
       </Badge>
   )}
   ```

2. **Agregar Métricas de Rendimiento**:
   - Enviar timestamps de inicio/fin de cada paso
   - Calcular FPS en frontend
   - Mostrar en `MainHeader`

3. **Optimizar Conversión**:
   - Solo convertir región visible (ROI)
   - Lazy loading de coordenadas fuera de pantalla
   - Cachear conversiones para coordenadas sin cambios

## Comandos Útiles

### Verificar si atheria_core está disponible:
```python
python3 -c "import atheria_core; print('✅ Nativo disponible')"
```

### Exportar modelo manualmente a JIT:
```python
python3 scripts/export_model_to_jit.py
```

### Forzar uso de motor Python:
```python
# En src/config.py
USE_NATIVE_ENGINE = False
```

