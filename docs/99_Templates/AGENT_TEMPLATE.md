---
tags: template, agent
---

# Plantilla de Definición de Agente de IA

**Instrucciones:** Copia este archivo a `docs/30_Components/AGENT_NOMBRE.md` y rellena la información.

---

# 🤖 Agente: [Nombre del Agente]

| Propiedad | Valor |
| :--- | :--- |
| **Nombre Completo** | `[Nombre Completo del Agente]` |
| **Identidad (Alias)** | `[Alias, ej: Gemini CLI, Cursor, Jules]` |
| **Archivo de Mandamientos Principal** | `ruta/al/archivo/de/reglas.md` |
| **Proveedor** | `[ej: Google, OpenAI, Anthropic, Local]` |

## 1. Rol y Responsabilidades

*   **Rol Principal:** [Describe en una frase el rol principal del agente. Ej: "Desarrollo interactivo en la terminal para refactorización, testing y documentación."]
*   **Responsabilidades Clave:**
    *   [Responsabilidad 1, ej: "Seguir rigurosamente las convenciones del proyecto."]
    *   [Responsabilidad 2, ej: "Mantener la base de conocimientos (`docs/`) actualizada."]
    *   [Responsabilidad 3, ej: "Realizar commits frecuentes y atómicos."]

## 2. Fortalezas y Casos de Uso

*   **Ideal para:**
    *   [Caso de uso 1, ej: "Arreglar bugs específicos con tests de verificación."]
    *   [Caso de uso 2, ej: "Generar documentación para componentes nuevos."]
    *   [Caso de uso 3, ej: "Ejecutar scripts y comandos en el entorno local."]
*   **No ideal para:**
    *   [Anti-patrón 1, ej: "Tareas de refactorización masivas y asíncronas (usar Jules para eso)."]
    *   [Anti-patrón 2, ej: "Análisis visual de interfaces de usuario."]

## 3. Reglas Específicas y Overrides

*   [Regla específica 1, si aplica. Ej: "Este agente debe solicitar confirmación antes de ejecutar cualquier comando `git push`."]
*   [Regla específica 2, si aplica.]

## 4. Recomendaciones de Interacción

*   [Consejo 1, ej: "Para mejores resultados, proporcionar peticiones claras y atómicas."]
*   [Consejo 2, ej: "Utilizar los comandos `/` definidos en el `AGENT_TOOLKIT` para acciones estandarizadas."]
