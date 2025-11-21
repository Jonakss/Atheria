---
title: Tags de Versión en Commits
type: template
tags: [git, versioning, ci/cd]
created: 2025-01-XX
---

# 🏷️ Tags de Versión en Mensajes de Commit

**Propósito:** Activar bump automático de versión cuando se hacen commits directos a `main` o `master`.

---

## 📋 Formato

Incluye uno de estos tags al final del mensaje de commit:

- `[version:bump:major]` - Incrementa versión mayor (X.0.0)
- `[version:bump:minor]` - Incrementa versión menor (0.X.0)
- `[version:bump:patch]` - Incrementa versión patch (0.0.X)

---

## ✅ Ejemplos

### Bump Patch (Corrección de bugs)
```bash
git commit -m "fix: corregir error en FPS calculation [version:bump:patch]"
git commit -m "fix(backend): manejar error CUDA OOM [version:bump:patch]"
```

### Bump Minor (Nueva funcionalidad)
```bash
git commit -m "feat: implementar shaders WebGL [version:bump:minor]"
git commit -m "feat(frontend): agregar timeline viewer [version:bump:minor]"
```

### Bump Major (Cambio breaking)
```bash
git commit -m "refactor: cambiar protocolo WebSocket (breaking) [version:bump:major]"
git commit -m "feat: nueva API incompatible [version:bump:major]"
```

---

## ⚠️ Reglas

1. **El tag debe estar al final del mensaje** (después de la descripción)
2. **Usa formato consistente**: `[version:bump:major/minor/patch]`
3. **Si NO incluyes el tag**, el workflow NO hará bump (se salta silenciosamente)
4. **Solo funciona en commits directos a `main` o `master`**

---

## 🔄 Workflow

1. Haces commit con tag de versión
2. Push a `main` o `master`
3. GitHub Actions detecta el tag
4. Ejecuta bump automático de versión
5. Crea commit de versión
6. Crea tag Git
7. Crea release GitHub

---

## 📝 Cuándo Usar

### `[version:bump:patch]`
- Correcciones de bugs
- Hotfixes
- Mejoras menores
- Documentación

### `[version:bump:minor]`
- Nuevas funcionalidades
- Mejoras de rendimiento
- Nuevas visualizaciones
- Nuevos componentes

### `[version:bump:major]`
- Cambios incompatibles
- Refactorizaciones mayores
- Cambios de protocolo
- Breaking changes en API

---

## 🎯 Alternativas

Si prefieres NO usar tags en commits:

1. **Usar PRs con labels** (recomendado para colaboración)
2. **Usar workflow manual** desde GitHub Actions UI
3. **Usar script local**: `python scripts/bump_version.py --type patch`

---

*Última actualización: 2025-01-XX*

