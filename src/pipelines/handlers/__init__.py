"""Módulos de handlers para pipeline_server."""
# Re-exportar handlers por categoría para facilitar imports
from .experiment_handlers import HANDLERS as EXPERIMENT_HANDLERS
from .simulation_handlers import HANDLERS as SIMULATION_HANDLERS
from .inference_handlers import HANDLERS as INFERENCE_HANDLERS
from .analysis_handlers import HANDLERS as ANALYSIS_HANDLERS
from .visualization_handlers import HANDLERS as VISUALIZATION_HANDLERS
from .config_handlers import HANDLERS as CONFIG_HANDLERS
from .system_handlers import HANDLERS as SYSTEM_HANDLERS

__all__ = [
    'EXPERIMENT_HANDLERS',
    'SIMULATION_HANDLERS',
    'INFERENCE_HANDLERS',
    'ANALYSIS_HANDLERS',
    'VISUALIZATION_HANDLERS',
    'CONFIG_HANDLERS',
    'SYSTEM_HANDLERS',
]

