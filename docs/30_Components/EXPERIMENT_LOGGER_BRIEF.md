Componente: Experiment Logger (Auto-Doc)

Objetivo: Automatizar la generación de reportes de experimentos en Markdown para mantener un historial científico sin intervención manual.

Funcionalidad

El ExperimentLogger (src/utils/experiment_logger.py) actúa como un escriba que:

Crea archivos en docs/40_Experiments/{Nombre_Experimento}.md.

Registra la configuración inicial (hiperparámetros).

Añade filas a una tabla de resultados cada vez que se guarda un checkpoint significativo.

Integración Requerida

Todos los entrenadores (QC_Trainer_v3, QC_Trainer_v4, etc.) deben:

Instanciar self.doc_logger = ExperimentLogger(name) en su __init__.

Llamar a self.doc_logger.initialize_or_load(config) al inicio.

Llamar a self.doc_logger.log_result(...) dentro de save_checkpoint cuando is_best=True o al final de un ciclo.