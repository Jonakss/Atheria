# Common Bugs & Solutions

Database de bugs encontrados durante el desarrollo de Atheria, con sus soluciones documentadas para referencia futura del RAG.

---

## 🐛 Backend Bugs

### SyntaxError: Dictionary Not Closed

**Fecha:** 2025-11-26  
**Severidad:** 🔴 CRITICAL (Server no inicia)  
**Archivo:** `src/managers/history_manager.py`

**Síntoma:**
```
SyntaxError: '{' was never closed
```

**Causa Raíz:**
Dictionary return statement sin cerrar bracket + código duplicado:

```python
# ❌ INCORRECTO
return {
    'total_frames': len(self.frames),
    'min_step': first_frame['step'],
# Falta cerrar bracket
first_frame = self.frames[0]  # Código duplicado
```

**Solución:**
```python
# ✅ CORRECTO
return {
    'total_frames': len(self.frames),
    'min_step': first_frame['step'],
    'max_step': last_frame['step']
}
```

**Prevención:**
- Usar editor con bracket matching
- Lint automático antes de commit
- Code review

**Referencias:** Commit `c149283`

---

## 🎨 Frontend Bugs

### Z-Index Overlap in Timeline Controls

**Fecha:** 2025-11-26  
**Severidad:** 🟡 MEDIUM (UI broken pero no blocking)  
**Archivo:** `frontend/src/modules/History/HistoryControls.tsx`

**Síntoma:**
Timeline slider appears behind other UI elements (PhysicsInspector, panels, etc.)

**Causa Raíz:**
No z-index defined on HistoryControls container, browser uses default stacking order.

**Solución:**
```tsx
// ❌ INCORRECTO
<div className="flex flex-col gap-3 p-4 bg-gray-800/50 rounded-lg border border-gray-700">

// ✅ CORRECTO
<div className="flex flex-col gap-3 p-4 bg-gray-800/50 rounded-lg border border-gray-700 relative z-10">
```

**Prevención:**
- Siempre definir z-index explícitamente para overlays/controls
- Usar z-index conventions:
  - `z-0`: Base layer
  - `z-10`: UI elements (buttons, forms)
  - `z-20`: Dropdowns, tooltips
  - `z-30`: Modals
  - `z-40`: Notifications
  - `z-50`: Critical alerts

**Referencias:** Commit `63aed21`

---

## 🔧 Build Bugs

### TypeScript Version Warning

**Fecha:** Recurrente  
**Severidad:** 🟢 LOW (Warning only, no runtime impact)  
**Archivo:** General build output

**Síntoma:**
```
WARNING: You are currently running a version of TypeScript which is not officially supported by @typescript-eslint/typescript-estree.
```

**Causa:**
TypeScript 5.9.3 not yet officially supported by eslint plugin.

**Solución:**
- **Opción A:** Ignorar (no afecta funcionalidad)
- **Opción B:** Downgrade TypeScript a última versión soportada
- **Opción C:** Esperar actualización de @typescript-eslint

**Status:** Aceptado como warning conocido.

---

## 📋 Bug Reporting Template

Cuando encuentres un bug nuevo, documéntalo aquí usando este template:

```markdown
### [Nombre Descriptivo del Bug]

**Fecha:** YYYY-MM-DD  
**Severidad:** 🔴 CRITICAL / 🟡 MEDIUM / 🟢 LOW  
**Archivo:** `path/to/file`

**Síntoma:**
[Describe qué se observa]

**Causa Raíz:**
[Explica POR QUÉ ocurre el bug]

**Solución:**
```code
# Muestra antes y después
```

**Prevención:**
[Cómo evitar este bug en el futuro]

**Referencias:** Commit hash o PR
```

---

## 🔗 Ver También

- [[TESTING_GUIDELINES]] - Best practices para detectar bugs temprano
- [[AI_DEV_LOG]] - Log cronológico de bugs encontrados
- [[TROUBLESHOOTING]] - Guía general de troubleshooting
