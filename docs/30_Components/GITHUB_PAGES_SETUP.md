# Configuración de GitHub Pages

## ⚙️ Habilitar GitHub Pages (Una sola vez)

GitHub Pages necesita ser habilitado manualmente en la configuración del repositorio:

### Pasos:

1. **Ve a tu repositorio** en GitHub:
   ```
   https://github.com/Jonakss/Atheria
   ```

2. **Click en "Settings"** (⚙️ en la barra superior)

3. **En el menú lateral izquierdo, click en "Pages"**

4. **En "Build and deployment":**
   - **Source**: Selecciona **"GitHub Actions"** (NO "Deploy from a branch")
   - Aparecerá un mensaje confirmando la configuración

5. **Click "Save"**

6. **Espera 1-2 minutos** y recarga la página. Deberías ver:
   ```
   Your site is published at https://jonakss.github.io/Atheria/
   ```

---

## 🚀 Después de la Configuración

Una vez habilitado, **cada push a `main`** que modifique archivos de `frontend/` automáticamente:

1. ✅ Ejecuta `npm run build` en el frontend
2. ✅ Sube el `dist/` a GitHub Pages
3. ✅ Despliega en `https://jonakss.github.io/Atheria/`

**No necesitas hacer nada más.** El workflow `deploy-pages.yml` se encarga de todo.

---

## 🔍 Verificar el Estado

Para ver si el deploy fue exitoso:

1. Ve a la pestaña **"Actions"** en tu repo
2. Busca el workflow **"Deploy Frontend to GitHub Pages"**
3. Debería aparecer ✅ verde si funcionó correctamente

URL del sitio:
```
https://jonakss.github.io/Atheria/
```

---

## ⚠️ Troubleshooting

### Error: "Resource not accessible by integration"
- **Solución**: Asegúrate de haber configurado **Source: GitHub Actions** en Settings → Pages

### El sitio no se actualiza
- **Solución**: Verifica que haya cambios en `frontend/` en tu último commit
- El workflow solo se ejecuta si hay cambios en frontend

### 404 Not Found en la URL
- **Solución**: Espera 1-2 minutos después del primer deploy
- Verifica que el workflow haya terminado correctamente (Actions tab)

---

## 📋 Resumen

| Configuración | Valor |
|--------------|--------|
| **URL del sitio** | https://jonakss.github.io/Atheria/ |
| **Source** | GitHub Actions |
| **Workflow** | `.github/workflows/deploy-pages.yml` |
| **Trigger** | Push a `main` con cambios en `frontend/` |
| **Build** | Vite + React |
| **Output** | `frontend/dist/` |
