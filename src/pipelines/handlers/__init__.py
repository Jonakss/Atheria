"""Módulos de handlers para pipeline_server."""
# Re-exportar handlers por categoría para facilitar imports
from .experiment_handlers import HANDLERS as EXPERIMENT_HANDLERS
from .simulation_handlers import HANDLERS as SIMULATION_HANDLERS
from .inference_handlers import HANDLERS as INFERENCE_HANDLERS
from .system_handlers import HANDLERS as SYSTEM_HANDLERS

# Placeholders para módulos que aún no existen (se crearán progresivamente)
# ANALYSIS_HANDLERS, VISUALIZATION_HANDLERS, CONFIG_HANDLERS se agregarán cuando se extraigan

__all__ = [
    'EXPERIMENT_HANDLERS',
    'SIMULATION_HANDLERS',
    'INFERENCE_HANDLERS',
    'SYSTEM_HANDLERS',
]

