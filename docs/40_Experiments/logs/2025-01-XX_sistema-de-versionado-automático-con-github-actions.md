## 2025-01-XX - Sistema de Versionado Automático con GitHub Actions

### Contexto
Para mantener sincronizadas las versiones en todos los componentes del proyecto (Backend Python, Motor Nativo C++, Frontend React) y automatizar el proceso de release, se implementó un sistema de versionado automático usando GitHub Actions.

### Problema Resuelto

#### Antes
- Versiones manuales en múltiples archivos
- Riesgo de inconsistencias entre componentes
- Proceso de release manual y propenso a errores
- No había trazabilidad automática de versiones

#### Después
- ✅ Versionado automático sincronizado en todos los componentes
- ✅ Uso de labels en PRs para determinar bump de versión (major/minor/patch)
- ✅ Creación automática de tags y releases
- ✅ Workflow manual disponible para bump manual si es necesario

### Implementación

#### 1. GitHub Actions Workflow

**Archivo:** `.github/workflows/version-bump.yml`

**Características:**
- Se ejecuta automáticamente cuando se hace merge a `main` o `master`
- También disponible como workflow manual (`workflow_dispatch`)
- Detecta labels en PRs para determinar tipo de bump
- Actualiza versiones en todos los archivos necesarios

#### 2. Labels de GitHub

**Labels requeridos para bump automático:**
- `version:major` o `major-version` o `breaking`: Incrementa versión mayor (X.0.0)
- `version:minor` o `minor-version` o `feature`: Incrementa versión menor (0.X.0)
- `version:patch` o `patch-version` o `bugfix` o `fix`: Incrementa versión patch (0.0.X)

**Por defecto:** Si no hay label, usa `patch` (más seguro)

#### 3. Archivos Actualizados Automáticamente

1. **`src/__version__.py`** (Fuente de verdad principal)
   - `__version__ = "X.Y.Z"`
   - `__version_info__ = (X, Y, Z)`

2. **`src/engines/__version__.py`**
   - `ENGINE_VERSION = "X.Y.Z"`

3. **`src/cpp_core/include/version.h`**
   - `ATHERIA_NATIVE_VERSION_MAJOR X`
   - `ATHERIA_NATIVE_VERSION_MINOR Y`
   - `ATHERIA_NATIVE_VERSION_PATCH Z`
   - `ATHERIA_NATIVE_VERSION_STRING "X.Y.Z"`

4. **`frontend/package.json`**
   - `"version": "X.Y.Z"`

#### 4. Proceso Automático

1. PR mergeado a `main` con label apropiado
2. Workflow detecta label y determina tipo de bump
3. Lee versión actual desde `src/__version__.py`
4. Calcula nueva versión según bump type
5. Actualiza todos los archivos de versión
6. Crea commit con mensaje: `chore: bump version to X.Y.Z [skip ci]`
7. Crea tag de Git: `vX.Y.Z`
8. Crea release de GitHub con descripción

#### 5. Workflow Manual

También disponible como workflow manual para bump manual:

```bash
# Desde GitHub Actions UI o API
# Opciones: major, minor, patch
```

### SemVer (Semantic Versioning)

**Formato:** `MAJOR.MINOR.PATCH`

- **MAJOR (X.0.0)**: Cambios incompatibles en la API
  - Cambios breaking en protocolos
  - Cambios incompatibles en configuraciones
  - Refactorizaciones mayores
  
- **MINOR (0.X.0)**: Nuevas funcionalidades compatibles hacia atrás
  - Nuevas features
  - Nuevos endpoints/APIs
  - Mejoras de rendimiento sin breaking changes
  
- **PATCH (0.0.X)**: Correcciones de bugs compatibles
  - Bugfixes
  - Correcciones de seguridad
  - Mejoras menores

### Uso

#### Para PRs (Automático)
1. Crear PR normalmente
2. Agregar label apropiado (`version:major`, `version:minor`, `version:patch`)
3. Hacer merge a `main`
4. Workflow se ejecuta automáticamente

#### Para Commits Directos (Agente/Desarrollo)
Cuando haces commits directos a `main`, incluye un tag de versión en el mensaje:

```bash
git commit -m "feat: nueva funcionalidad [version:bump:minor]"
git commit -m "fix: corrección de bug [version:bump:patch]"
git commit -m "refactor: cambio breaking [version:bump:major]"
```

**Tags disponibles:**
- `[version:bump:major]` - Bump mayor (X.0.0)
- `[version:bump:minor]` - Bump menor (0.X.0)
- `[version:bump:patch]` - Bump patch (0.0.X)

**Si NO incluyes el tag**, el workflow se salta silenciosamente (no hace bump).

#### Para Bump Manual
1. Ir a GitHub Actions → "Version Bump Automático"
2. Click en "Run workflow"
3. Seleccionar tipo de bump (major/minor/patch)
4. Ejecutar

### Notas

- El workflow requiere permisos `contents: write` y `pull-requests: write`
- Los commits de bump incluyen `[skip ci]` para evitar loops infinitos
- El workflow usa `GITHUB_TOKEN` automático (no requiere secrets adicionales)
- Todas las versiones se mantienen sincronizadas automáticamente

### Beneficios

- ✅ Sincronización automática de versiones
- ✅ Trazabilidad de releases
- ✅ Proceso reproducible y confiable
- ✅ Releases automáticos en GitHub
- ✅ Tags de Git para referencias específicas

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
