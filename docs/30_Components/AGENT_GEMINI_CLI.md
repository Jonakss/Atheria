---
tags: agent, MOC
---

# 🤖 Agente: Gemini CLI

| Propiedad | Valor |
| :--- | :--- |
| **Nombre Completo** | `Gemini Command Line Interface` |
| **Identidad (Alias)** | `Gemini CLI` |
| **Archivo de Mandamientos Principal** | `.gemini/GEMINI.md` |
| **Proveedor** | `Google` |

## 1. Rol y Responsabilidades

*   **Rol Principal:** Ser el ingeniero principal de IA para el desarrollo interactivo en la terminal, encargado de la implementación, refactorización, testing y mantenimiento de la base de conocimientos.
*   **Responsabilidades Clave:**
    *   Seguir y hacer cumplir rigurosamente las reglas definidas en el `Archivo de Mandamientos Principal`.
    *   Mantener la base de conocimientos (`docs/`) consultada y actualizada en cada paso.
    *   Realizar commits frecuentes, atómicos y bien documentados, incluyendo el versionado automático.
    *   Interactuar con el usuario para clarificar ambigüedades y planificar tareas complejas.

## 2. Fortalezas y Casos de Uso

*   **Ideal para:**
    *   **Desarrollo Iterativo:** Arreglar bugs, añadir features y escribir tests, todo dentro de un ciclo de feedback rápido.
    *   **Gestión de la Base de Conocimientos:** Crear, actualizar y enlazar documentación en `docs/` como parte del flujo de desarrollo.
    *   **Ejecución de Comandos y Scripts:** Utilizar la terminal para compilar, testear, y diagnosticar el estado del proyecto.
    *   **Refactorización Atómica:** Realizar cambios de código precisos y controlados en archivos específicos.
*   **No ideal para:**
    *   **Tareas a Gran Escala y Asíncronas:** Para refactorizaciones que abarcan todo el proyecto o análisis de dependencias complejos, es preferible usar un agente especializado como **Google Jules**.
    *   **Análisis Visual:** No puede interpretar interfaces gráficas o elementos visuales.

## 3. Reglas Específicas y Overrides

*   La regla **#4 (Documentación Viva)** es de cumplimiento **CRÍTICO** para este agente. Cada cambio de código debe ir acompañado de una consulta y/o actualización de la documentación.
*   Debe usar los comandos `/` definidos en `docs/99_Templates/AGENT_TOOLKIT.md` cuando el usuario los invoque.

## 4. Recomendaciones de Interacción

*   Proporcionar objetivos claros y, si es posible, divididos en subtareas.
*   Para tareas complejas, revisar y aprobar el plan propuesto por el agente antes de que comience la implementación.
*   Utilizar los comandos como `/new_experiment` o `/doc` para estandarizar operaciones repetitivas.
