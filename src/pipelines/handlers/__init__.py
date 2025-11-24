from .experiment_handlers import HANDLERS as EXPERIMENT_HANDLERS
from .simulation_handlers import HANDLERS as SIMULATION_HANDLERS
from .inference_handlers import HANDLERS as INFERENCE_HANDLERS
from .system_handlers import HANDLERS as SYSTEM_HANDLERS
from .analysis_handlers import HANDLERS as ANALYSIS_HANDLERS
from .history_handlers import HANDLERS as HISTORY_HANDLERS

# Agrupar todos los handlers por scope
HANDLERS = {
    "experiment": EXPERIMENT_HANDLERS,
    "simulation": SIMULATION_HANDLERS,
    "inference": INFERENCE_HANDLERS,
    "system": SYSTEM_HANDLERS,
    "analysis": ANALYSIS_HANDLERS,
    "history": HISTORY_HANDLERS
}

__all__ = [
    'HANDLERS',
    'EXPERIMENT_HANDLERS',
    'SIMULATION_HANDLERS',
    'INFERENCE_HANDLERS',
    'SYSTEM_HANDLERS',
    'ANALYSIS_HANDLERS',
    'HISTORY_HANDLERS'
]
