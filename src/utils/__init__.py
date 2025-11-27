"""
Utilidades para Atheria 4.

Este módulo re-exporta funciones del archivo src/utils.py para mantener compatibilidad.
"""

import sys
from pathlib import Path
import importlib.util

# Cargar directamente el archivo utils.py (no el módulo src.utils para evitar recursión)
src_dir = Path(__file__).parent.parent
utils_py_path = src_dir / "utils.py"

if not utils_py_path.exists():
    raise ImportError(f"No se encontró {utils_py_path}")

# Cargar el módulo con un nombre único para evitar conflictos
spec = importlib.util.spec_from_file_location("_src_utils_py", utils_py_path)
_utils_py_module = importlib.util.module_from_spec(spec)

# Necesitamos establecer el contexto del paquete para que los imports relativos funcionen
# Establecer sys.modules['src'] si no existe
if 'src' not in sys.modules:
    # Crear un módulo dummy para src
    import types
    src_module = types.ModuleType('src')
    sys.modules['src'] = src_module

# Establecer el contexto del paquete para que los imports relativos funcionen
_utils_py_module.__package__ = 'src'

# Ejecutar el módulo (esto ejecutará todos los imports)
# Necesitamos establecer sys.modules['src'] y sys.modules['src.config'] antes
# de ejecutar para que los imports relativos funcionen
if 'src.config' not in sys.modules:
    # Importar config primero para que esté disponible cuando utils.py lo necesite
    import src.config
    sys.modules['src.config'] = src.config

spec.loader.exec_module(_utils_py_module)

# Re-exportar funciones desde _utils_py_module (archivo src/utils.py)
get_experiment_list = _utils_py_module.get_experiment_list
load_experiment_config = _utils_py_module.load_experiment_config
get_latest_checkpoint = _utils_py_module.get_latest_checkpoint
get_latest_jit_model = _utils_py_module.get_latest_jit_model
calculate_training_time_from_checkpoints = _utils_py_module.calculate_training_time_from_checkpoints
save_experiment_config = _utils_py_module.save_experiment_config
sns_to_dict_recursive = _utils_py_module.sns_to_dict_recursive
check_and_create_dir = _utils_py_module.check_and_create_dir

# Importar desde el directorio utils
from .experiment_logger import ExperimentLogger
from .binary_loader import (
    get_platform_info,
    get_atheria_version,
    download_prebuilt_binary,
    try_load_native_engine,
)

# Exportar todo
__all__ = [
    'ExperimentLogger',
    'get_experiment_list',
    'load_experiment_config',
    'get_latest_checkpoint',
    'get_latest_jit_model',
    'calculate_training_time_from_checkpoints',
    'save_experiment_config',
    'sns_to_dict_recursive',
    'check_and_create_dir',
    # Binary loader utilities
    'get_platform_info',
    'get_atheria_version',
    'download_prebuilt_binary',
    'try_load_native_engine',
]
