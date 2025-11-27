# Motor Nativo: Sistema de Releases Multi-Plataforma

Sistema automatizado de GitHub Actions para compilar y distribuir binarios pre-compilados del motor nativo C++ (`atheria_core`) en múltiples plataformas, eliminando la necesidad de compilación local en entornos como Kaggle/Co lab/Lightning AI.

## 🎯 Objetivo

Facilitar el uso del motor nativo de alto rendimiento en entornos de cloud computing donde compilar C++ es complejo o consume mucho tiempo de cuota de GPU.

## 🏗️ Arquitectura

### Componentes

```
.github/workflows/native-engine-release.yml  # Workflow de compilación
src/utils/binary_loader.py                  # Auto-descarga en notebooks
notebooks/Atheria_Progressive_Training.ipynb # Integración en entrenamiento
```

### Flujo Automático

```mermaid
graph LR
    A[Commit con tag versión] --> B[version-bump.yml]
    B --> C[Crea Release v4.2.x]
    C --> D[native-engine-release.yml]
    D --> E[Build Matrix 3 OS × 2 Python]
    E --> F[Upload 6 binaries to Release]
    F --> G[Disponible para descarga]
```

## 📦 Binarios Generados

Cada release incluye 6 archivos:

| Plataforma | Python | Archivo |
|-----------|--------|---------|
| Linux x64 | 3.10 | `atheria_core-linux-x86_64-py310.so` |
| Linux x64 | 3.11 | `atheria_core-linux-x86_64-py311.so` |
| macOS arm64 | 3.10 | `atheria_core-macos-arm64-py310.so` |
| macOS arm64 | 3.11 | `atheria_core-macos-arm64-py311.so` |
| Windows x64 | 3.10 | `atheria_core-windows-x64-py310.pyd` |
| Windows x64 | 3.11 | `atheria_core-windows-x64-py311.pyd` |

## 🚀 Uso en Notebooks

### Auto-descarga (Recomendado)

El notebook `Atheria_Progressive_Training.ipynb` incluye auto-descarga:

```python
from src.utils.binary_loader import try_load_native_engine

# Intenta importar atheria_core, si falla descarga automáticamente
if try_load_native_engine(fallback_to_download=True):
    print("✅ Motor nativo disponible")
else:
    print("⚠️ Usando motor Python (más lento pero funcional)")
```

### Descarga Manual

Si necesitas descargar manualmente:

```python
from src.utils.binary_loader import download_prebuilt_binary
from pathlib import Path

# Descargar binario para la versión actual
binary_path = download_prebuilt_binary(
    version="4.2.6",  # None = auto-detectar
    target_dir=Path("src/"),
    verbose=True
)
```

### Plataforma Detectada

```python
from src.utils.binary_loader import get_platform_info

info = get_platform_info()
print(f"OS: {info['os']}")        # linux, macos, windows
print(f"Arch: {info['arch']}")    # x86_64, arm64
print(f"Python: {info['python_version']}")  # 310, 311
```

## 🔧 Workflow de Build

### Trigger

El workflow se ejecuta automáticamente cuando:

```yaml
on:
  release:
    types: [created]  # Cuando version-bump.yml crea un release
  workflow_dispatch:   # Manual desde GitHub Actions UI
```

### Build Matrix

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ['3.10', '3.11']
```

Genera **6 jobs en paralelo** (3 OS × 2 Python versions).

### Pasos Clave

1. **Setup**: Checkout, Python, PyTorch (LibTorch)
2. **Build**: `python setup.py build_ext --inplace`
3. **Identify**: Buscar `*atheria_core*.{so,pyd}`
4. **Rename**: `atheria_core-{platform}-py{version}.{ext}`
5. **Upload**: Agregar al release como asset

## 📊 Ventajas

| Aspecto | Sin Releases | Con Releases |
|---------|-------------|-------------|
| **Tiempo de setup** | ~10-15 min (compilar) | ~30 seg (descargar) |
| **Dependencias** | CMake, pybind11, compilers | Solo requests |
| **Cuota GPU** | Se consume compilando | Solo para entrenar |
| **Compatibilidad** | Puede fallar por entorno | Testeado en CI |
| **Almacenamiento** | ~500MB (build artifacts) | ~50MB (binario) |

## 🛠️ Troubleshooting

### Error 404 al descargar

**Causa**: No existe binario para tu plataforma/Python version.

**Solución**:
1. Verificar plataforma: `python -c "from src.utils.binary_loader import get_platform_info; print(get_platform_info())"`
2. Verificar releases: https://github.com/Jonakss/Atheria/releases/latest
3. Compilar manualmente: `python setup.py build_ext --inplace`

### Import Error después de descargar

**Causa**: Incompatibilidad de versión de PyTorch o Python.

**Solución**:
```python
# Verificar versión de PyTorch
import torch
print(torch.__version__)  # Debe ser compatible con CI (CPU version)

# Verificar Python
import sys
print(sys.version_info)  # Debe ser 3.10 o 3.11
```

### Binario descargado pero no funciona

**Causa**: Posible corrupción o incompatibilidad de LibTorch.

**Solución**:
```bash
# Re-descargar
rm src/atheria_core*.so
python -c "from src.utils.binary_loader import download_prebuilt_binary; download_prebuilt_binary()"

# Verificar integridad
ls -lh src/atheria_core*.so  # Debe ser ~10-50MB

# Fallback: usar motor Python
# (el trainer detecta automáticamente si atheria_core no está disponible)
```

## 🔄 Actualización Manual de Binarios

### Crear Release Manualmente

```bash
# 1. Tag local
git tag -a v4.2.6 -m "Release 4.2.6"
git push origin v4.2.6

# 2. Crear release en GitHub UI
# - Ir a Releases → Draft new release
# - Tag: v4.2.6
# - Publicar

# 3. El workflow se ejecuta automáticamente
# 4. Verificar en Actions tab
```

### Trigger Manual del Workflow

```bash
# Desde GitHub UI:
# 1. Ir a Actions → Build Native Engine Releases
# 2. Run workflow
# 3. Ingresar versión: 4.2.6
# 4. Run
```

## 🌐 URLs de Descarga

Formato:
```
https://github.com/Jonakss/Atheria/releases/download/v{version}/atheria_core-{platform}-py{pyver}.{ext}
```

Ejemplos:
```
https://github.com/Jonakss/Atheria/releases/download/v4.2.6/atheria_core-linux-x86_64-py311.so
https://github.com/Jonakss/Atheria/releases/download/v4.2.6/atheria_core-windows-x64-py310.pyd
```

## 📝 Notas

- **LibTorch**: Los binarios incluyen enlaces a LibTorch (CPU version)
- **Tamaño**: ~10-50MB por binario (puede variar según configuración)
- **CI Runners**: GitHub-hosted runners (Ubuntu 22.04, macOS 14, Windows Server 2022)
- **Caché**: Los binarios se cacheán en `src/` después de la primera descarga

## 🔗 Referencias

- [[PROGRESSIVE_TRAINING_GUIDE]] - Guía de uso en notebooks
- [[VERSIONING_SYSTEM]] - Sistema de versionado automático
- Workflow: [native-engine-release.yml](file:///home/jonathan.correa/Projects/Atheria/.github/workflows/native-engine-release.yml)
- Loader: [binary_loader.py](file:///home/jonathan.correa/Projects/Atheria/src/utils/binary_loader.py)
