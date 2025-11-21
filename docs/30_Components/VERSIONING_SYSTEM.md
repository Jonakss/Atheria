---
title: Sistema de Versionado Automático
type: component
status: implemented
tags: [ci/cd, versioning, automation, github-actions]
created: 2025-01-XX
updated: 2025-01-XX
related: [[40_Experiments/AI_DEV_LOG|AI Dev Log]]
---

# 🔢 Sistema de Versionado Automático

**Objetivo:** Mantener sincronizadas las versiones en todos los componentes del proyecto y automatizar el proceso de release usando GitHub Actions.

---

## 📋 Componentes con Versión

El proyecto Atheria 4 mantiene versiones en múltiples componentes:

1. **Backend Python** (`src/__version__.py`)
   - Versión principal de la aplicación
   
2. **Motor Python** (`src/engines/__version__.py`)
   - Versión del motor de simulación Python y wrapper nativo
   
3. **Motor Nativo C++** (`src/cpp_core/include/version.h`)
   - Versión del motor nativo C++ compilado con LibTorch
   
4. **Frontend React** (`frontend/package.json`)
   - Versión de la aplicación frontend

**Todas las versiones se mantienen sincronizadas automáticamente.**

---

## 🔄 Workflow de Versionado

### Activación Automática

El workflow se ejecuta automáticamente cuando:

1. **PR mergeado a `main` o `master`**
   - Revisa labels del PR para determinar tipo de bump
   - Actualiza versiones automáticamente
   - Crea tag y release

2. **Workflow Manual** (`workflow_dispatch`)
   - Permite bump manual desde GitHub Actions UI
   - Selección manual de tipo de bump

### Archivo: `.github/workflows/version-bump.yml`

---

## 🏷️ Labels para Bump de Versión

Para que el workflow detecte automáticamente el tipo de bump, los PRs deben tener uno de estos labels:

### Major Version (X.0.0)
**Labels:**
- `version:major`
- `major-version`
- `breaking`

**Cuándo usar:**
- Cambios incompatibles en la API
- Cambios breaking en protocolos WebSocket
- Cambios incompatibles en configuraciones
- Refactorizaciones mayores que rompen compatibilidad

**Ejemplo:** `4.1.0` → `5.0.0`

### Minor Version (0.X.0)
**Labels:**
- `version:minor`
- `minor-version`
- `feature`

**Cuándo usar:**
- Nuevas funcionalidades
- Nuevos endpoints/APIs (compatibles hacia atrás)
- Nuevos tipos de visualización
- Mejoras de rendimiento sin breaking changes

**Ejemplo:** `4.1.0` → `4.2.0`

### Patch Version (0.0.X)
**Labels:**
- `version:patch`
- `patch-version`
- `bugfix`
- `fix`

**Cuándo usar:**
- Correcciones de bugs
- Correcciones de seguridad
- Mejoras menores
- Optimizaciones internas

**Ejemplo:** `4.1.0` → `4.1.1`

### Por Defecto
Si no hay label explícito, se usa `patch` automáticamente (más seguro).

---

## 📝 Proceso Automático

Cuando se mergea un PR con label apropiado:

1. **Detectar tipo de bump** desde labels del PR
2. **Leer versión actual** desde `src/__version__.py` (fuente de verdad)
3. **Calcular nueva versión** según bump type
4. **Actualizar archivos:**
   - `src/__version__.py`
   - `src/engines/__version__.py`
   - `src/cpp_core/include/version.h`
   - `frontend/package.json`
5. **Crear commit:** `chore: bump version to X.Y.Z [skip ci]`
6. **Push commit** a la rama principal
7. **Crear tag Git:** `vX.Y.Z`
8. **Crear release GitHub** con descripción del PR

---

## 🚀 Uso Manual

Para hacer bump manual desde GitHub Actions:

1. Ir a **Actions** → **Version Bump Automático**
2. Click en **Run workflow**
3. Seleccionar tipo de bump:
   - `major`: Incrementa versión mayor
   - `minor`: Incrementa versión menor
   - `patch`: Incrementa versión patch
4. Click en **Run workflow**

---

## 📊 SemVer (Semantic Versioning)

Atheria 4 sigue [Semantic Versioning](https://semver.org/):

**Formato:** `MAJOR.MINOR.PATCH`

- **MAJOR (X.0.0)**: Cambios incompatibles
- **MINOR (0.X.0)**: Nuevas funcionalidades compatibles
- **PATCH (0.0.X)**: Correcciones de bugs

### Ejemplos

- `4.1.0` → `5.0.0`: Cambio breaking (ej: cambio de protocolo WebSocket)
- `4.1.0` → `4.2.0`: Nueva feature (ej: nuevos tipos de visualización)
- `4.1.0` → `4.1.1`: Bugfix (ej: corrección de error en FPS)

---

## 📂 Archivos de Versión

### Backend Python
**Archivo:** `src/__version__.py`
```python
__version__ = "4.1.0"
__version_info__ = (4, 1, 0)
```

### Motor Python/Wrapper
**Archivo:** `src/engines/__version__.py`
```python
ENGINE_VERSION = "4.1.0"
```

### Motor Nativo C++
**Archivo:** `src/cpp_core/include/version.h`
```cpp
#define ATHERIA_NATIVE_VERSION_MAJOR 4
#define ATHERIA_NATIVE_VERSION_MINOR 1
#define ATHERIA_NATIVE_VERSION_PATCH 0
#define ATHERIA_NATIVE_VERSION_STRING "4.1.0"
```

### Frontend React
**Archivo:** `frontend/package.json`
```json
{
  "version": "4.0.2"
}
```

**Nota:** Frontend puede tener versión diferente (independiente), pero se actualiza automáticamente cuando se hace bump.

---

## 🔗 Referencias

- [[40_Experiments/AI_DEV_LOG|AI Dev Log]] - Log de desarrollo con detalles
- [Semantic Versioning](https://semver.org/) - Especificación SemVer
- `.github/workflows/version-bump.yml` - Workflow de GitHub Actions

---

*Última actualización: 2025-01-XX*

