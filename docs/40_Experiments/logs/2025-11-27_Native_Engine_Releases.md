## 2025-11-27 - System: Multi-Platform Native Engine Releases

### Contexto
Implementación de sistema automatizado de GitHub Actions para compilar y distribuir binarios pre-compilados del motor nativo C++ (`atheria_core`) en múltiples plataformas.

### Motivación
- **Kaggle/Colab sin compilación**: Eliminar necesidad de compilar C++ localmente (ahorra ~10-15 min de setup)
- **Cuota GPU optimizada**: No desperdiciar cuota compilando, solo entrenar
- **Soporte multi-plataforma**: Linux, macOS, Windows × Python 3.10/3.11
- **Binarios testeados en CI**: Garantizan compatibilidad

### Arquitectura Implementada

#### 1. GitHub Actions Workflow ✅

**Archivo:** `.github/workflows/native-engine-release.yml`

**Build Matrix:**
```yaml
matrix:
  os: [ubuntu-latest, macos-latest, windows-latest]
  python-version: ['3.10', '3.11']
```

**Trigger:** Automático al crear release (integrado con `version-bump.yml`)

**Proceso:**
1. Setup Python + PyTorch (LibTorch)
2. Compilar: `python setup.py build_ext --inplace`
3. Detectar binario: `*atheria_core*.{so,pyd}`
4. Renombrar: `atheria_core-{platform}-py{version}.{ext}`
5. Upload a GitHub Releases como asset

**Binarios generados por release:**
- `atheria_core-linux-x86_64-py310.so`
- `atheria_core-linux-x86_64-py311.so`
- `atheria_core-macos-arm64-py310.so`
- `atheria_core-macos-arm64-py311.so`
- `atheria_core-windows-x64-py310.pyd`
- `atheria_core-windows-x64-py311.pyd`

#### 2. Binary Loader Utility ✅

**Archivo:** `src/utils/binary_loader.py`

**Funciones principales:**
```python
get_platform_info() -> dict  # Detecta OS, arch, Python version
download_prebuilt_binary(version, target_dir) -> Path  # Descarga desde GitHub
try_load_native_engine(fallback_to_download=True) -> bool  # Auto-carga o descarga
```

**Flujo de auto-descarga:**
1. Intenta `import atheria_core`
2. Si falla → detecta plataforma
3. Construye URL: `https://github.com/Jonakss/Atheria/releases/download/v{version}/atheria_core-{platform}-py{pyver}.{ext}`
4. Descarga a `src/` con nombre compatible con import
5. Re-intenta importar

**Fallback gracioso:** Si falla descarga, continúa con motor Python (más lento pero funcional)

### Decisiones de Diseño

#### ¿Por qué releases en vez de PyPI wheels?
- **LibTorch**: Los wheels con LibTorch son enormes (~500MB), exceden límites de PyPI
- **CI más simple**: GitHub Releases tienen límites más generosos (2GB/asset)
- **Flexibilidad**: Podemos distribuir solo binarios, no todo el package

#### ¿Por qué auto-descarga en notebooks?
- **UX simplificada**: Usuario no necesita descargar manualmente
- **Versionado automático**: Descarga versión correcta del release
- **Detección de plataforma**: Selecciona binario correcto automáticamente

### Ventajas

| Aspecto | Sin Releases | Con Releases |
|---------|-------------|-------------|
| Tiempo setup | ~10-15 min compilar | ~30 seg descargar |
| Dependencias | CMake, pybind11, gcc | Solo requests |
| Cuota GPU | Se consume compilando | Solo entrenar |
| Compatibilidad | Puede fallar por entorno | Testeado en CI |

### Archivos Creados/Modificados

**Workflows:**
- `.github/workflows/native-engine-release.yml` - Build matrix (NUEVO)

**Python:**
- `src/utils/binary_loader.py` - Auto-descarga (NUEVO)
- `src/utils/__init__.py` - Exports de binary_loader

**Documentación:**
- `docs/30_Components/NATIVE_ENGINE_RELEASES.md` - Arquitectura completa (NUEVO)
- `docs/40_Experiments/AI_DEV_LOG.md` - Esta entrada

### Uso en Notebooks

El notebook progresivo incluirá integración:

```python
from src.utils.binary_loader import try_load_native_engine

# Auto-descarga si no está disponible
if try_load_native_engine(fallback_to_download=True):
    print("✅ Motor nativo disponible (aceleración 10-50x)")
else:
    print("⚠️ Usando motor Python (funcional pero más lento)")
```

### Limitaciones Conocidas

⚠️ **Primera vez**: El workflow se ejecutará en el próximo release (no hay binarios históricos)
⚠️ **Tamaño**: Binarios ~10-50MB (depende de LibTorch linkeo)
⚠️ **macOS ARM**: Solo para Apple Silicon (M1/M2/M3)

### Referencias

- [[NATIVE_ENGINE_RELEASES]] - Documentación completa
- [[PROGRESSIVE_TRAINING_GUIDE]] - Uso en notebooks
- `setup.py` - Build system con CMake
- `.github/workflows/version-bump.yml` - Versionado automático
