## 2024-12-20 - Corrección Segfault: Cleanup Motor Nativo

### Contexto
Se detectó un **segmentation fault (core dumped)** al cargar un experimento después de que se hubiera inicializado el motor nativo C++. El segfault ocurría cuando:

1. El motor nativo C++ se inicializaba primero (por ejemplo, al verificar disponibilidad)
2. Luego se decidía usar el motor Python
3. El motor nativo no se limpiaba correctamente antes de crear el motor Python
4. Al destruir el wrapper del motor nativo, los recursos C++ se liberaban de forma incorrecta

**Error observado:**
```
🚀 MOTOR NATIVO LISTO: device=cuda, grid_size=256
🐍 MOTOR PYTHON ACTIVO: device=cuda, grid_size=256
...
Segmentation fault (core dumped)
```

### Causa Raíz
El `NativeEngineWrapper` no tenía un método explícito de cleanup. Cuando Python hacía garbage collection del wrapper:

1. El destructor de Python (`__del__`) no liberaba explícitamente el motor nativo C++
2. Los tensores PyTorch en `state.psi` podían tener referencias circulares
3. El motor nativo C++ (`atheria_core.Engine`) se destruía después de que sus dependencias ya habían sido liberadas
4. Esto causaba acceso a memoria inválida → segfault

### Solución Implementada

#### 1. Método `cleanup()` Explícito

**Archivo:** `src/engines/native_engine_wrapper.py`

Se agregó un método `cleanup()` que libera recursos de forma controlada:

```python
def cleanup(self):
    """
    Limpia recursos del motor nativo de forma explícita.
    Debe llamarse antes de destruir el wrapper para evitar segfaults.
    """
    # Limpiar estado denso primero
    if hasattr(self, 'state') and self.state is not None:
        if hasattr(self.state, 'psi') and self.state.psi is not None:
            self.state.psi = None
        self.state = None
    
    # Limpiar referencias al motor nativo
    if hasattr(self, 'native_engine') and self.native_engine is not None:
        self.native_engine = None
    
    # Limpiar otras referencias
    self.model_loaded = False
    self.step_count = 0
    self.last_delta_psi = None
    ...
```

**Orden de cleanup:**
1. Primero: liberar tensores PyTorch (estado denso)
2. Segundo: liberar motor nativo C++ (cuando no hay dependencias)
3. Tercero: limpiar otras referencias

#### 2. Destructor Mejorado

Se agregó `__del__()` que llama a `cleanup()` automáticamente:

```python
def __del__(self):
    """Destructor - llama a cleanup para asegurar limpieza correcta."""
    try:
        self.cleanup()
    except Exception:
        # Ignorar errores en destructor para evitar problemas durante GC
        pass
```

#### 3. Cleanup Explícito en `handle_load_experiment`

**Archivo:** `src/pipelines/pipeline_server.py`

Se mejoró el cleanup del motor anterior antes de crear uno nuevo:

```python
# CRÍTICO: Limpiar motor nativo explícitamente antes de eliminarlo
if hasattr(old_motor, 'native_engine'):
    if hasattr(old_motor, 'cleanup'):
        old_motor.cleanup()
        logging.debug("Motor nativo limpiado explícitamente antes de eliminarlo")
```

#### 4. Cleanup al Fallar Inicialización

Cuando el motor nativo falla al inicializarse o cargar el modelo, se limpia correctamente:

```python
temp_motor = NativeEngineWrapper(...)
try:
    if temp_motor.load_model(jit_path):
        motor = temp_motor
        temp_motor = None  # Evitar cleanup - motor se usará
    else:
        # Limpiar motor nativo que falló
        if temp_motor is not None:
            temp_motor.cleanup()
            temp_motor = None
except Exception as e:
    # Limpiar motor nativo que falló durante inicialización
    if temp_motor is not None:
        temp_motor.cleanup()
        temp_motor = None
```

### Justificación

**Por qué cleanup explícito:**
- **Seguridad:** Evita segfaults por destrucción incorrecta de objetos C++
- **Predecibilidad:** Orden de destrucción controlado
- **Debugging:** Más fácil identificar problemas de memoria

**Por qué usar variable temporal:**
- Permite limpiar el motor nativo incluso si falla la carga del modelo
- Evita asignar a `motor` hasta que esté completamente inicializado
- Reduce riesgo de referencias colgantes

### Archivos Modificados

1. **`src/engines/native_engine_wrapper.py`**
   - Agregado método `cleanup()`
   - Agregado destructor `__del__()`

2. **`src/pipelines/pipeline_server.py`**
   - Mejorado cleanup del motor anterior en `handle_load_experiment`
   - Agregado cleanup cuando el motor nativo falla

### Testing

**Validación:**
- ✅ Cargar experimento con motor Python después de inicializar motor nativo
- ✅ Cambiar de motor nativo a Python sin segfault
- ✅ Motor nativo falla durante inicialización → cleanup correcto
- ✅ Motor nativo falla al cargar modelo → cleanup correcto

**Pruebas recomendadas:**
- Cargar múltiples experimentos consecutivamente
- Alternar entre motores nativo y Python
- Forzar fallos durante inicialización

### Estado
✅ **Completado y probado**

**Referencias:**
- [[Native_Engine_Core#Cleanup y Gestión de Memoria]]
- `src/engines/native_engine_wrapper.py:407-442`
- `src/pipelines/pipeline_server.py:1019-1042`

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
