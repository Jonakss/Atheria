🧰 Atheria Agent Toolkit (Comandos)

Este archivo define "Macro-Comandos" para estandarizar las tareas repetitivas del Agente en Cursor.
Cuando el usuario invoque un comando (ej: /new_experiment), sigue las instrucciones asociadas rigurosamente.

🧪 Comandos de Ciencia

/new_experiment

Trigger: Crear un nuevo experimento de entrenamiento.
Acción:

Preguntar al usuario: Nombre, Arquitectura (V3/V4) y Objetivo.

Crear la carpeta output/experiments/{NAME}.

Generar el archivo de configuración inicial.

Instanciar el ExperimentLogger en docs/40_Experiments/{NAME}.md.

/log_result

Trigger: Registrar resultados de un benchmark o entrenamiento.
Acción:

Leer el archivo docs/40_Experiments/{CURRENT_EXP}.md.

Si no existe, crearlo usando docs/99_Templates/Experiment_Log_Template.md.

Agregar una nueva fila a la tabla de "Resultados" con fecha, métricas y notas.

No borrar historial anterior.

/epoch_check

Trigger: Verificar en qué Era Cosmológica estamos.
Acción:

Ejecutar el script src/analysis/epoch_detector.py sobre el último snapshot.

Reportar al usuario: "Estamos en la Era X (Simetría: Y, Energía: Z)".

💻 Comandos de Ingeniería

/refactor

Trigger: Solicitar limpieza de código.
Acción:

Leer docs/AGENT_GUIDELINES.md sección "Filosofía de Código".

Revisar el archivo actual buscando:

Bucles for en Python que deberían ser vectorizados.

Type hints faltantes.

Comentarios desactualizados.

Proponer el código refactorizado.

/cpp_bridge

Trigger: Crear o actualizar bindings C++.
Acción:

Verificar que la función existe en C++ (src/cpp_core).

Verificar que está expuesta en bindings.cpp.

Verificar que Python puede llamarla en native_engine.py.

Si algo falta, generar el código de pegamento (glue code).

📚 Comandos de Documentación

/doc

Trigger: Documentar un archivo nuevo.
Acción:

Analizar el código del archivo abierto.

Generar un archivo Markdown en docs/30_Components/ con el mismo nombre.

Usar la plantilla docs/99_Templates/Component_Template.md.

Rellenar Inputs, Outputs y Lógica automáticamente.

/roadmap_update

Trigger: Marcar una tarea como completada.
Acción:

Leer docs/10_Core/ROADMAP_PHASE_1.md.

Marcar con [x] la tarea mencionada.

Si todas las tareas de una sección están listas, sugerir pasar a la siguiente Fase.