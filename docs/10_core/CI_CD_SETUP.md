# Configuración de GitHub Actions

Atheria utiliza GitHub Actions y el CLI de Gemini para automatizar tareas como revisión de código, triaje de issues e invocación de comandos AI.

Este documento explica cómo configurar el repositorio para que estas acciones funcionen correctamente.

## 🔑 Secretos y Variables Requeridos

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
| `GEMINI_MODEL` | `gemini-1.5-flash` | El modelo a usar. `gemini-1.5-flash` es rápido y gratuito (con límites). `gemini-1.5-pro` es más potente. | ✅ SÍ |
| `GOOGLE_GENAI_USE_VERTEXAI` | `false` | Ponlo en `false` si usas **AI Studio** (API Key). Ponlo en `true` si usas **Vertex AI**. | ✅ SÍ |
| `GOOGLE_GENAI_USE_GCA` | `false` | Uso de Gemini Code Assist (opcional). | No |

## 🚀 Uso de los Comandos

Una vez configurado, puedes usar los siguientes comandos en comentarios de Issues o Pull Requests:

- **`@gemini-cli /triage`**: Analiza un issue y le asigna etiquetas (labels) automáticamente.
- **`@gemini-cli /review`**: (En un PR) Realiza una revisión de código detallada.
- **`@gemini-cli /invoke [prompt]`**: Ejecuta una instrucción personalizada.
  - Ejemplo: `@gemini-cli /invoke Explícame qué hace el archivo qca_engine.py`

## 🛠 Solución de Problemas

- **Error: "Resource has been exhausted"**: Si usas la capa gratuita de AI Studio, es posible que alcances el límite de cuota. Espera unos minutos o cambia a un modelo más ligero (`gemini-1.5-flash`).
- **El bot no responde**: Asegúrate de que los workflows tienen permisos de lectura/escritura en **Settings** -> **Actions** -> **General** -> **Workflow permissions**.

## Referencias

- [Google AI Studio](https://aistudio.google.com/)
- [Gemini CLI Action](https://github.com/google-github-actions/run-gemini-cli)
