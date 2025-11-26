# 🗺️ MOC de Reglas de Agentes de IA

**Última actualización:** 2025-11-26

## 1. Propósito

Este documento es el **Map of Content (MOC)** y la fuente central de verdad para todos los agentes de IA que operan en el proyecto Atheria 4. Su objetivo es:
- **Centralizar:** Definir qué agentes están activos.
- **Estandarizar:** Proporcionar un punto de acceso único a las reglas y mandamientos de cada agente.
- **Facilitar la Actualización:** Permitir a los agentes (y a los humanos) saber qué archivo de reglas deben modificar cuando se les solicita.

## 2. Flujo de Trabajo para Agentes

**CUANDO SE TE PIDA ACTUALIZAR TUS REGLAS O MANDAMIENTOS:**
1.  **Consulta este archivo** para identificar tu identidad.
2.  **Navega al archivo de definición** (ej: `[[AGENT_GEMINI_CLI]]`).
3.  **Localiza tu `Archivo de Mandamientos Principal`**.
4.  **Modifica únicamente ese archivo**.

**CUANDO SE INTRODUCE UN NUEVO AGENTE:**
1.  **Añade el nuevo agente** a la lista de abajo.
2.  **Crea un nuevo archivo de definición** en `docs/30_Components/` usando la plantilla `[[AGENT_TEMPLATE]]`.
3.  **Enlaza el nuevo archivo** desde esta página.

---

## 3. Agentes Activos

A continuación se listan todos los agentes de IA aprobados para trabajar en este proyecto.

| Agente | Archivo de Definición | Archivo de Mandamientos Principal | Rol Principal |
| :--- | :--- | :--- | :--- |
| 🤖 **Gemini CLI** | [[AGENT_GEMINI_CLI]] | `.gemini/GEMINI.md` | Desarrollo interactivo en terminal |
| 🚀 **Google Jules** | `Próximamente` | `N/A` | Tareas asíncronas a gran escala |
| 👁️ **Cursor** | `Próximamente` | `N/A` | Asistencia en el editor de código |
| ⚡ **Lightning AI** | `Próximamente` | `N/A` | Gestión de infraestructura de entrenamiento |
| 🌌 **Antigravity** | `Próximamente` | `N/A` | Análisis de código y dependencias |

*(Esta tabla debe ser actualizada por cualquier agente al que se le notifique de un nuevo colega o de su propia incorporación).*
