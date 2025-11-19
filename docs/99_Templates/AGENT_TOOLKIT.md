# 🧰 Atheria Agent Toolkit (Comandos)

Este archivo define "Macro-Comandos" para estandarizar las tareas repetitivas del Agente en Cursor.
Cuando el usuario invoque un comando (ej: `/new_experiment`), sigue las instrucciones asociadas rigurosamente.

---

## 🧪 Comandos de Ciencia

### `/new_experiment`
**Trigger:** Crear un nuevo experimento de entrenamiento.
**Acción:**
1.  Preguntar al usuario: Nombre, Arquitectura (V3/V4) y Objetivo.
2.  Crear la carpeta `output/experiments/{NAME}`.
3.  Generar el archivo de configuración inicial.
4.  Instanciar el `ExperimentLogger` en `docs/40_Experiments/{NAME}.md`.

### `/log_result`
**Trigger:** Registrar resultados de un benchmark o entrenamiento.
**Acción:**
1.  Leer el archivo `docs/40_Experiments/{CURRENT_EXP}.md`.
2.  Si no existe, crearlo usando `docs/99_Templates/Experiment_Log_Template.md`.
3.  Agregar una nueva fila a la tabla de "Resultados" con fecha, métricas y notas.
4.  No borrar historial anterior.

### `/epoch_check`
**Trigger:** Verificar en qué Era Cosmológica estamos.
**Acción:**
1.  Ejecutar el script `src/analysis/epoch_detector.py` sobre el último snapshot.
2.  Reportar al usuario: "Estamos en la Era X (Simetría: Y, Energía: Z)".

---

## 💻 Comandos de Ingeniería

### `/refactor`
**Trigger:** Solicitar limpieza de código.
**Acción:**
1.  Leer `docs/99_Templates/AGENT_GUIDELINES.md` sección "Filosofía de Código".
2.  Revisar el archivo actual buscando:
    - Bucles `for` en Python que deberían ser vectorizados.
    - Type hints faltantes.
    - Comentarios desactualizados.
3.  Proponer el código refactorizado.

### `/cpp_bridge`
**Trigger:** Crear o actualizar bindings C++.
**Acción:**
1.  Verificar que la función existe en C++ (`src/cpp_core`).
2.  Verificar que está expuesta en `bindings.cpp`.
3.  Verificar que Python puede llamarla en `native_engine.py`.
4.  Si algo falta, generar el código de pegamento (glue code).

---

## 📚 Comandos de Documentación

### `/doc`
**Trigger:** Documentar un archivo nuevo.
**Acción:**
1.  Analizar el código del archivo abierto.
2.  Generar un archivo Markdown en `docs/30_Components/` con el mismo nombre.
3.  Usar la plantilla `docs/99_Templates/Component_Template.md`.
4.  Rellenar Inputs, Outputs y Lógica automáticamente.

### `/roadmap_update`
**Trigger:** Marcar una tarea como completada.
**Acción:**
1.  Leer `docs/10_Core/ROADMAP_PHASE_1.md`.
2.  Marcar con `[x]` la tarea mencionada.
3.  Si todas las tareas de una sección están listas, sugerir pasar a la siguiente Fase.

IMPORTANTE: Puedes ejecutar macro-comandos definidos en docs/99_Templates/AGENT_TOOLKIT.md cuando el usuario escriba /comando (ej: /doc, /refactor). Lee ese archivo para saber qué pasos seguir.