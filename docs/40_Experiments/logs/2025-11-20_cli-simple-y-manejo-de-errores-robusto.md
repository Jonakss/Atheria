## 2025-11-20 - CLI Simple y Manejo de Errores Robusto

### Contexto
Creación de un CLI simple para facilitar el flujo de desarrollo y mejoras en el manejo de errores para prevenir segfaults y errores de conversión de tipos.

### Problemas Resueltos

#### 1. Comando Largo para Desarrollo
- **Antes:** `python3 setup.py build_ext --inplace && pip install -e . && ATHERIA_NO_FRONTEND=1 python3 run_server.py`
- **Después:** `atheria dev` o `python3 src/cli.py dev`

#### 2. Errores de Conversión de Tipos
- **Antes:** `'numpy.ndarray' object has no attribute 'detach'` cuando se intentaba convertir arrays numpy como si fueran tensores PyTorch
- **Después:** Verificaciones robustas con `isinstance()` y `hasattr()`, con fallback a `np.array()`

#### 3. Segfaults al Cambiar de Engine
- **Antes:** Segmentation fault al cambiar de motor nativo a Python sin cleanup adecuado
- **Después:** Cleanup explícito del motor anterior antes de cambiar, con try-except robusto

### Implementación

#### 1. CLI Simple (`src/cli.py`)

**Comandos disponibles:**
- `atheria dev` - Build + Install + Run (sin frontend por defecto)
- `atheria dev --frontend` - Build + Install + Run (con frontend)
- `atheria build` - Solo compilar extensiones C++
- `atheria install` - Solo instalar paquete
- `atheria run` - Solo ejecutar servidor
- `atheria clean` - Limpiar archivos de build

**Características:**
- Manejo de comandos con `argparse`
- Ejecución de comandos con `subprocess`
- Mensajes claros con emojis para mejor UX
- Manejo de errores con try-except

**Entry Points en `setup.py`:**
```python
entry_points={
    'console_scripts': [
        'atheria=src.cli:main',
        'ath=src.cli:main',  # Alias corto
    ],
}
```

#### 2. Manejo Robusto de Conversión de Tipos

**Archivo:** `src/pipelines/pipeline_viz.py`

**Cambios:**
- Cada conversión (density, phase, real_part, imag_part, energy) tiene su propio try-except
- Verifica `isinstance(tensor, torch.Tensor)` Y `hasattr(tensor, 'detach')` antes de llamar `.detach()`
- Fallback a `np.array()` si falla la conversión

**Resultado:**
- ✅ No más errores de `'numpy.ndarray' object has no attribute 'detach'`
- ✅ Manejo robusto de objetos híbridos o tipos inesperados

#### 3. Cleanup Robusto al Cambiar Engine

**Archivo:** `src/pipelines/pipeline_server.py`

**Función:** `handle_switch_engine()`

**Cambios:**
- Cleanup explícito del motor anterior ANTES de cambiar
- Try-except alrededor de todas las operaciones de cleanup
- Verificaciones con `hasattr()` antes de acceder a atributos

**Resultado:**
- ✅ No más segfaults al cambiar de motor nativo a Python
- ✅ Cleanup robusto incluso si hay errores

### Archivos Modificados
1. **`src/cli.py`** (nuevo) - CLI completo
2. **`setup.py`** - Agregado `entry_points`
3. **`src/pipelines/pipeline_viz.py`** - Manejo robusto de conversión
4. **`src/pipelines/pipeline_server.py`** - Cleanup robusto en switch_engine

### Estado
✅ **Completado**

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
