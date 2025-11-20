# Problemas de Rendimiento Motor Nativo

**Fecha:** 2024-12-20  
**Estado:** 🔴 **CRÍTICO - Pendiente de Resolución**

---

## 🐛 Problemas Identificados

### 1. Motor Nativo se Cuelga/Bloquea

**Síntoma:**
- El motor nativo se queda bloqueado durante la simulación
- No responde a comandos de pausa inmediatamente
- Requiere matar el proceso para detener

**Causa Raíz:**
1. **`step_native()` es bloqueante**: La función C++ `step_native()` procesa todo el batch sin verificar pausa
2. **Conversión disperso→denso bloqueante**: `_update_dense_state_from_sparse()` se ejecuta en cada paso y puede tomar mucho tiempo con grids grandes
3. **Sin verificación de pausa durante ejecución**: Aunque se verifica antes de llamar `evolve_internal_state()`, no se verifica durante la ejecución

**Ubicación del Problema:**
- `src/cpp_core/src/sparse_engine.cpp:71` - `step_native()` es bloqueante
- `src/engines/native_engine_wrapper.py:283` - `_update_dense_state_from_sparse()` se ejecuta en cada paso
- `src/pipelines/pipeline_server.py:257` - No hay verificación de pausa durante `evolve_internal_state()`

---

### 2. Lentitud Extrema en Tiempo Real

**Síntoma:**
- El motor nativo se pone muy lento en tiempo real
- FPS cae dramáticamente
- UI se congela

**Causa Raíz:**
1. **Conversión completa en cada paso**: `_update_dense_state_from_sparse()` itera sobre **todo el grid** (256x256 = **65,536 coordenadas**) en cada paso
2. **Llamadas individuales a `get_state_at()`**: Para cada coordenada se hace una llamada al motor nativo C++ (overhead de Python↔C++)
3. **No hay optimización con ROI**: Convierte todo el grid incluso si solo se necesita una región
4. **No hay lazy conversion**: Convierte incluso si `live_feed` está desactivado

**Análisis de Complejidad:**
- Grid 256x256 = 65,536 coordenadas
- En cada paso: 65,536 llamadas a `get_state_at()`
- Overhead Python↔C++ × 65,536 = **MUY COSTOSO**

**Ubicación del Problema:**
- `src/engines/native_engine_wrapper.py:285-372` - `_update_dense_state_from_sparse()`
- `src/pipelines/pipeline_server.py:257` - Se ejecuta en cada `evolve_internal_state()`

---

## 🔧 Soluciones Propuestas

### Solución 1: Lazy Conversion (Prioridad Alta)

**Objetivo:** Solo convertir cuando se necesita visualizar

**Implementación:**
```python
def evolve_internal_state(self):
    """Evoluciona el estado interno usando el motor nativo."""
    if not self.model_loaded:
        return
    
    # Ejecutar paso nativo (todo en C++)
    particle_count = self.native_engine.step_native()
    self.step_count += 1
    
    # NO convertir aquí - solo marcar como "stale"
    self._dense_state_stale = True

def get_dense_state(self):
    """Obtiene el estado denso, convirtiendo solo si es necesario."""
    if self._dense_state_stale or self.state.psi is None:
        self._update_dense_state_from_sparse()
        self._dense_state_stale = False
    return self.state.psi
```

**Beneficios:**
- Solo convierte cuando se necesita (al enviar frame)
- Puede saltarse conversión si `live_feed` está desactivado
- Reduce overhead en pasos que no se visualizan

---

### Solución 2: Usar ROI (Region of Interest) para Conversión Parcial

**Objetivo:** Solo convertir región visible

**Implementación:**
```python
def _update_dense_state_from_sparse(self, roi=None):
    """Convierte solo la región de interés si se proporciona."""
    if roi is None:
        # Sin ROI: convertir todo (fallback)
        roi_coords = [(x, y) for y in range(self.grid_size) 
                      for x in range(self.grid_size)]
    else:
        # Con ROI: solo convertir región visible
        x_min, y_min, x_max, y_max = roi
        roi_coords = [(x, y) for y in range(y_min, y_max)
                      for x in range(x_min, x_max)]
    
    # Convertir solo coordenadas en ROI
    for x, y in roi_coords:
        coord = atheria_core.Coord3D(x, y, 0)
        state_tensor = self.native_engine.get_state_at(coord)
        self.state.psi[0, y, x] = state_tensor.to(self.device)
```

**Beneficios:**
- Reduce conversión de 65,536 a ~10,000-20,000 coordenadas (si ROI es pequeño)
- Puede ser 3-5x más rápido dependiendo del tamaño de ROI
- Mejor experiencia de usuario al hacer zoom/pan

---

### Solución 3: Batch Conversion en C++ (Prioridad Media)

**Objetivo:** Reducir overhead Python↔C++

**Implementación:**
- Agregar método C++ `get_state_batch()` que obtiene múltiples coordenadas en una llamada
- Procesar en batches de 1000-5000 coordenadas

**Beneficios:**
- Reduce overhead de llamadas Python↔C++
- Puede ser 10-50x más rápido que llamadas individuales

---

### Solución 4: Verificación de Pausa Durante Conversión

**Objetivo:** Permitir pausa inmediata durante conversión

**Implementación:**
```python
def _update_dense_state_from_sparse(self, check_pause_callback=None):
    """Convierte con verificación de pausa periódica."""
    BATCH_SIZE = 1000
    
    coords_list = [...]
    for i in range(0, len(coords_list), BATCH_SIZE):
        # Verificar pausa cada batch
        if check_pause_callback and check_pause_callback():
            logging.debug("Conversión interrumpida por pausa")
            return  # Salir temprano
        
        batch_coords = coords_list[i:i+BATCH_SIZE]
        # Procesar batch...
```

**Beneficios:**
- Permite pausa inmediata incluso durante conversión
- No bloquea UI durante conversión larga

---

### Solución 5: Cache de Estado Denso (Prioridad Baja)

**Objetivo:** Reutilizar conversión si estado no cambió

**Implementación:**
- Usar hash del estado disperso para detectar cambios
- Solo convertir si estado cambió significativamente

**Beneficios:**
- Evita conversión innecesaria si estado no cambió
- Útil cuando hay muchos frames sin cambios

---

## 📊 Priorización

### 🔴 Crítico (Implementar Inmediatamente)
1. **Lazy Conversion** - Solo convertir cuando se necesita
2. **ROI para Conversión** - Solo convertir región visible

### 🟡 Alto (Implementar Pronto)
3. **Verificación de Pausa Durante Conversión** - Permitir pausa inmediata
4. **Batch Conversion en C++** - Reducir overhead Python↔C++

### 🟢 Medio (Implementar Después)
5. **Cache de Estado Denso** - Optimización adicional

---

## 🧪 Testing

### Pruebas Necesarias
1. **Test de Cuelgue:**
   - Iniciar simulación con motor nativo
   - Intentar pausar inmediatamente
   - Verificar que responde en < 1 segundo

2. **Test de Rendimiento:**
   - Medir tiempo de conversión antes/después
   - Comparar FPS con motor Python
   - Verificar que ROI mejora rendimiento

3. **Test de Lazy Conversion:**
   - Ejecutar 1000 pasos sin visualizar
   - Verificar que no hay conversión
   - Luego activar visualización y verificar conversión única

---

## 📝 Referencias

- [[Native_Engine_Core]] - Documentación del motor nativo
- [[NATIVE_ENGINE_COMMUNICATION]] - Comunicación Python↔C++
- `src/engines/native_engine_wrapper.py:285-372` - Conversión disperso→denso
- `src/cpp_core/src/sparse_engine.cpp:71` - Función `step_native()`

---

**Estado:** 📋 **Documentado - Pendiente de Implementación**

