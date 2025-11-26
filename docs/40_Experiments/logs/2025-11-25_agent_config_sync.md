# Sincronización de Configuración de Agentes

**Fecha:** 2025-11-25
**Autor:** Antigravity (Agent)
**Estado:** Completado

## Descripción
Se ha implementado una configuración unificada para agentes de IA en el entorno de Lightning AI, alineándola con las configuraciones existentes de Cursor y Gemini. Además, se ha actualizado la documentación central para garantizar que todos los agentes tengan conocimiento de estas configuraciones y se mantengan sincronizados.

## Cambios Realizados

1.  **Creación de `.lightning`:**
    - Se creó un nuevo archivo de reglas en la raíz del proyecto con el mismo contenido que `.cursorrules` (en español).
    - Esto asegura que los agentes operando en el contexto de Lightning AI tengan las mismas directivas y "mandamientos" que en otros entornos.

2.  **Actualización de `AGENTS.md`:**
    - Se añadió una sección "AGENT CONFIGURATION SYNC".
    - Se listan explícitamente los archivos de configuración (`.cursorrules`, `.gemini/GEMINI.md`, `.lightning`) como fuentes de verdad que deben mantenerse en sincronía.

3.  **Actualización de `docs/99_Templates/AGENT_GUIDELINES.md`:**
    - Se añadió una sección equivalente en español para guiar a los agentes que consulten las plantillas.

## Propósito
El objetivo es evitar la fragmentación del comportamiento de los agentes. Independientemente de la herramienta utilizada (Cursor, Gemini, Lightning Studio), el agente debe seguir los mismos protocolos de documentación, estilo de código y gestión de estado definidos en el proyecto Atheria 4.
