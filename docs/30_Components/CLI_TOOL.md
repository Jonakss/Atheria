# CLI Tool - Atheria

**Fecha:** 2025-11-20  
**Componente:** `src/cli.py`

## Descripción

CLI simple para facilitar el flujo de desarrollo de Atheria 4. Simplifica comandos comunes como build, install y run.

## Uso

### Instalación

Después de instalar el paquete en modo desarrollo:
```bash
pip install -e .
```

El comando `atheria` estará disponible globalmente.

### Comandos Disponibles

#### `atheria -q "query"` / `atheria --query "query"`
Buscar en la Knowledge Base (`docs/`).

**Ejemplo:**
```bash
atheria -q "Harmonic Engine"
ath -q "quantum tools"
```

Muestra los documentos más relevantes con su ruta y un snippet del contenido.

#### `atheria dev`
Build + Install + Run (workflow completo de desarrollo).

**Opciones:**
- `--frontend` - Incluir frontend (por defecto: sin frontend)
- `--port PORT` - Puerto del servidor
- `--host HOST` - Host del servidor

**Ejemplos:**
```bash
atheria dev                  # Build + Install + Run (sin frontend)
atheria dev --frontend       # Build + Install + Run (con frontend)
atheria dev --port 8080      # Build + Install + Run en puerto 8080
```

#### `atheria build`
Solo compilar extensiones C++.

**Ejemplo:**
```bash
atheria build
```

#### `atheria install`
Solo instalar paquete en modo desarrollo.

**Ejemplo:**
```bash
atheria install
```

#### `atheria run`
Solo ejecutar servidor.

**Opciones:**
- `--frontend` - Incluir frontend (por defecto: sin frontend)
- `--port PORT` - Puerto del servidor
- `--host HOST` - Host del servidor

**Ejemplos:**
```bash
atheria run                  # Ejecutar servidor (sin frontend)
atheria run --frontend       # Ejecutar servidor (con frontend)
```

#### `atheria frontend-dev [--port PORT]`
Ejecuta solo el frontend en modo desarrollo (npm run dev).

**Opciones:**
- `--port PORT` - Puerto del frontend (por defecto: 5173)

**Ejemplos:**
```bash
atheria frontend-dev              # Frontend en modo desarrollo (puerto 5173)
atheria frontend-dev --port 3000  # Frontend en puerto 3000
```

**Nota:** Este comando ejecuta `npm run dev` en el directorio `frontend/`. Si `node_modules` no existe, ejecuta `npm install` automáticamente.

#### `atheria clean`
Limpiar archivos de build, cache y archivos temporales.

**Ejemplo:**
```bash
atheria clean
```

Limpia:
- `build/`
- `dist/`
- `*.egg-info/`
- `**/__pycache__/`
- `**/*.pyc`
- `**/*.pyo`
- `*.so`
- `**/*.so`
- `.pytest_cache/`
- `.mypy_cache/`

### Uso Directo (Sin Instalar)

También se puede usar directamente sin instalar:
```bash
python3 src/cli.py dev
python3 src/cli.py frontend-dev
python3 src/cli.py build
python3 src/cli.py install
python3 src/cli.py run
python3 src/cli.py clean
```

## Equivalencia de Comandos

### Antes
```bash
python3 setup.py build_ext --inplace && pip install -e . && ATHERIA_NO_FRONTEND=1 python3 run_server.py
```

### Ahora
```bash
atheria dev
```

## Alias

El comando `ath` también está disponible como alias corto:
```bash
ath dev
ath frontend-dev
ath build
ath install
ath run
ath clean
```

## Implementación

**Archivo:** `src/cli.py`

**Entry Points en `setup.py`:**
```python
entry_points={
    'console_scripts': [
        'atheria=src.cli:main',
        'ath=src.cli:main',  # Alias corto
    ],
}
```

## Características

- ✅ Manejo de comandos con `argparse`
- ✅ Ejecución de comandos con `subprocess`
- ✅ Mensajes claros con emojis para mejor UX
- ✅ Manejo de errores con try-except
- ✅ Detección automática del directorio del proyecto

## Relacionado

- [[AI_DEV_LOG#2025-11-20 - CLI Simple y Manejo de Errores Robusto]]

