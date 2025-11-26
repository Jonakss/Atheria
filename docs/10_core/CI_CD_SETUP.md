# Configuración de CI/CD y Automatización

Atheria utiliza GitHub Actions para dos propósitos principales:

1.  **CI/CD (Integración y Despliegue Continuo):** Automatización de pruebas y despliegue del proyecto.
2.  **Automatización con IA (Gemini CLI):** Ayuda en tareas de desarrollo como revisión de código y triaje de issues.

Este documento explica la configuración y funcionamiento de ambos sistemas.

## 1. Workflows de CI/CD del Proyecto

Estos flujos de trabajo aseguran la calidad del código y automatizan el despliegue del frontend.

### `ci.yml` - Integración Continua

Este workflow es el **guardián de la calidad del código**.

-   **Disparadores:** Se ejecuta automáticamente en cada `push` y `pull request` a la rama `main`.
-   **Objetivo:** Verificar que los nuevos cambios no rompen ninguna parte del proyecto.

**Proceso que ejecuta:**
1.  **Checkout:** Descarga el código del repositorio.
2.  **Setup Entornos:** Configura los entornos de Python (3.10) y Node.js (18).
3.  **Frontend Check:**
    - Instala dependencias (`npm ci`).
    - Valida el estilo del código (`npm run lint`).
    - Construye el proyecto para producción (`npm run build`) para asegurar que compila.
4.  **Backend Check:**
    - Instala dependencias de Python (`pip install -e .`).
    - Compila las extensiones nativas de C++ (`setup.py build_ext`).
    - Ejecuta la suite de pruebas del backend (`pytest`).

Si alguno de estos pasos falla, el workflow marcará el commit o PR como fallido.

**✨ Nueva Funcionalidad: Reporte de Errores en PRs**
Si el paso "Build Frontend" (`npm run build`) falla durante la ejecución de un Pull Request, el workflow publicará automáticamente un comentario en el PR con el log del error. Esto permite a los desarrolladores diagnosticar y corregir problemas de compilación rápidamente sin necesidad de revisar los logs completos del workflow.

### `deploy-pages.yml` - Despliegue a GitHub Pages

Este workflow es el **publicador automático del frontend**.

-   **Disparadores:** Se ejecuta automáticamente solo cuando se hace un `push` a la rama `main`.
-   **Objetivo:** Desplegar la última versión del frontend a GitHub Pages.

**Proceso que ejecuta:**
1.  **Checkout y Setup:** Descarga el código y configura Node.js.
2.  **Build Frontend:** Instala dependencias con `npm ci` y construye la versión de producción (`npm run build`).
3.  **Deploy:** Sube los archivos generados (del directorio `frontend/dist`) a GitHub Pages.

#### ⚠️ Acción Requerida

Para que este despliegue funcione, un administrador del repositorio debe hacer lo siguiente **una única vez**:
1.  Ir a **Settings** -> **Pages**.
2.  En la sección "Build and deployment", cambiar la **Source** a **"GitHub Actions"**.

---

## 2. Automatización con IA (Gemini CLI)

Atheria utiliza el CLI de Gemini para automatizar tareas como revisión de código, triaje de issues e invocación de comandos AI.

## 🔑 Secretos y Variables Requeridos (Gemini)

Para que los workflows de Gemini (`.github/workflows/gemini-*.yml`) funcionen, necesitas configurar los siguientes secretos y variables en tu repositorio.

Ve a **Settings** -> **Secrets and variables** -> **Actions**.

### Secrets (Repository Secrets)

| Nombre | Descripción | Requerido |
|--------|-------------|-----------|
| `GEMINI_API_KEY` | Tu API Key de Google AI Studio o Google Cloud Vertex AI. | ✅ SÍ |
| `GITHUB_TOKEN` | Generado automáticamente por GitHub Actions. | (Automático) |

### Variables (Repository Variables)

Ve a la pestaña **Variables** en la misma sección de configuración.

| Nombre | Valor Recomendado | Descripción | Requerido |
|--------|-------------------|-------------|-----------|
| `GEMINI_MODEL` | `gemini-2.0-flash` | El modelo a usar. Se recomienda **gemini-2.0-flash** (GA) por velocidad y costo. También puedes usar `gemini-2.0-pro-exp`, `gemini-2.5-flash` o `gemini-3.0-pro-preview` si tienes acceso. | ✅ SÍ |
| `GOOGLE_GENAI_USE_VERTEXAI` | `false` | Ponlo en `false` si usas **AI Studio** (API Key). Ponlo en `true` si usas **Vertex AI**. | ✅ SÍ |
| `GOOGLE_GENAI_USE_GCA` | `false` | Uso de Gemini Code Assist (opcional). | No |

## 🚀 Uso de los Comandos

Una vez configurado, puedes usar los siguientes comandos en comentarios de Issues o Pull Requests:

- **`@gemini-cli /triage`**: Analiza un issue y le asigna etiquetas (labels) automáticamente.
- **`@gemini-cli /review`**: (En un PR) Realiza una revisión de código detallada.
- **`@gemini-cli /invoke [prompt]`**: Ejecuta una instrucción personalizada.
  - Ejemplo: `@gemini-cli /invoke Explícame qué hace el archivo run_server.py`

## 🛠 Solución de Problemas

- **Error: "Resource has been exhausted"**: Si usas la capa gratuita de AI Studio, es posible que alcances el límite de cuota. Espera unos minutos o cambia a un modelo más ligero.
- **El bot no responde**: Asegúrate de que los workflows tienen permisos de lectura/escritura en **Settings** -> **Actions** -> **General** -> **Workflow permissions**.

## Referencias

- [Google AI Studio](https://aistudio.google.com/)
- [Gemini CLI Action](https://github.com/google-github-actions/run-gemini-cli)
- [Modelos Gemini Disponibles](https://ai.google.dev/gemini-api/docs/models)
