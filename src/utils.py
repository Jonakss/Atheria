# src/utils.py
"""
Utilidades para gestión de experimentos, configuración y archivos.
"""
import os
import json
import logging
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
import glob

from . import config as global_cfg


def get_experiment_list():
    """
    Escanea el directorio de experimentos y devuelve una lista de experimentos válidos.
    Ahora es robusto a ficheros config.json corruptos e incluye información sobre checkpoints.
    """
    experiments = []
    if not os.path.exists(global_cfg.EXPERIMENTS_DIR):
        return experiments

    for exp_name in os.listdir(global_cfg.EXPERIMENTS_DIR):
        exp_path = os.path.join(global_cfg.EXPERIMENTS_DIR, exp_name)
        config_path = os.path.join(exp_path, 'config.json')
        if os.path.isdir(exp_path) and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                    # --- ARREGLO DE COMPATIBILIDAD ---
                    # Asegura que MODEL_PARAMS exista, incluso en configs antiguas
                    if 'MODEL_PARAMS' not in config_data:
                        config_data['MODEL_PARAMS'] = {
                            'd_state': config_data.get('D_STATE', 8),
                            'hidden_channels': config_data.get('HIDDEN_CHANNELS', 32)
                        }
                    # --- FIN ARREGLO ---

                    # Verificar si tiene checkpoints disponibles (sin loguear warnings)
                    checkpoint_path = get_latest_checkpoint(exp_name, silent=True)
                    has_checkpoint = checkpoint_path is not None

                    # Obtener timestamps del config o del sistema de archivos
                    created_at = config_data.get('created_at')
                    updated_at = config_data.get('updated_at')
                    last_training_time = config_data.get('last_training_time')
                    total_training_time = calculate_training_time_from_checkpoints(exp_name)

                    # Devolver datos en formato compatible con el frontend
                    # El frontend espera config anidado, pero también datos planos para compatibilidad
                    experiments.append({
                        'name': exp_name,
                        # Datos planos (snake_case) para compatibilidad
                        'model_architecture': config_data.get('MODEL_ARCHITECTURE', 'N/A'),
                        'grid_size_training': config_data.get('GRID_SIZE_TRAINING', 'N/A'),
                        'qca_steps_training': config_data.get('QCA_STEPS_TRAINING', 'N/A'),
                        'total_episodes': config_data.get('TOTAL_EPISODES', 'N/A'),
                        'lr_rate_m': config_data.get('LR_RATE_M', 'N/A'),
                        'has_checkpoint': has_checkpoint,
                        'created_at': created_at,
                        'updated_at': updated_at,
                        'last_training_time': last_training_time,
                        'total_training_time': total_training_time,
                        # Config anidado (PascalCase) para compatibilidad con frontend
                        'config': {
                            'MODEL_ARCHITECTURE': config_data.get('MODEL_ARCHITECTURE', 'N/A'),
                            'GRID_SIZE_TRAINING': config_data.get('GRID_SIZE_TRAINING', 'N/A'),
                            'QCA_STEPS_TRAINING': config_data.get('QCA_STEPS_TRAINING', 'N/A'),
                            'TOTAL_EPISODES': config_data.get('TOTAL_EPISODES', 'N/A'),
                            'LR_RATE_M': config_data.get('LR_RATE_M', 'N/A'),
                            'MODEL_PARAMS': config_data.get('MODEL_PARAMS', {'d_state': 8, 'hidden_channels': 32}),
                            'GAMMA_DECAY': config_data.get('GAMMA_DECAY', 0.01),
                            'INITIAL_STATE_MODE_INFERENCE': config_data.get('INITIAL_STATE_MODE_INFERENCE', 'complex_noise'),
                            'LOAD_FROM_EXPERIMENT': config_data.get('LOAD_FROM_EXPERIMENT', None),
                            'D_STATE': config_data.get('MODEL_PARAMS', {}).get('d_state') if isinstance(config_data.get('MODEL_PARAMS'), dict) else config_data.get('D_STATE', 8),
                            'HIDDEN_CHANNELS': config_data.get('MODEL_PARAMS', {}).get('hidden_channels') if isinstance(config_data.get('MODEL_PARAMS'), dict) else config_data.get('HIDDEN_CHANNELS', 32),
                            # Información del motor y dispositivo
                            'TRAINING_DEVICE': config_data.get('TRAINING_DEVICE', 'cpu'),  # CPU o CUDA
                            'USE_NATIVE_ENGINE': config_data.get('USE_NATIVE_ENGINE', False),  # Si se intentó usar motor nativo
                        }
                    })
            except json.JSONDecodeError:
                logging.warning(f"Archivo config.json corrupto o inválido en '{exp_path}'. Ignorando.")
            except Exception as e:
                logging.error(f"Error al procesar experimento '{exp_name}': {e}", exc_info=True)

    # Ordenar por created_at, manejando None values correctamente
    # Si created_at es None, usar string vacío para ordenar al final
    experiments.sort(key=lambda x: x.get('created_at') or '', reverse=True)

    return experiments


def calculate_training_time_from_checkpoints(experiment_name: str) -> float:
    """
    Calcula el tiempo total de entrenamiento basado en los timestamps de los checkpoints.
    Retorna el tiempo en segundos.
    """
    checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, experiment_name)
    if not os.path.exists(checkpoint_dir):
        return 0.0

    try:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not checkpoints:
            return 0.0

        # Obtener timestamps de todos los checkpoints
        timestamps = []
        for checkpoint_file in checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            try:
                stat = os.stat(checkpoint_path)
                timestamps.append(stat.st_mtime)
            except:
                continue

        if len(timestamps) < 2:
            return 0.0

        # Calcular diferencia entre el primero y el último
        timestamps.sort()
        total_time = timestamps[-1] - timestamps[0]
        return total_time
    except Exception as e:
        logging.warning(f"Error calculando tiempo de entrenamiento para '{experiment_name}': {e}")
        return 0.0


def load_experiment_config(experiment_name: str):
    """
    Carga la configuración de un experimento específico y la devuelve como un SimpleNamespace.
    """
    config_path = os.path.join(global_cfg.EXPERIMENTS_DIR, experiment_name, 'config.json')
    if not os.path.exists(config_path):
        logging.error(f"No se encontró el fichero de configuración: {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

            # --- ARREGLO DE COMPATIBILIDAD ---
            # Asegura que MODEL_PARAMS exista, incluso en configs antiguas
            if 'MODEL_PARAMS' not in config_dict:
                config_dict['MODEL_PARAMS'] = {
                    'd_state': config_dict.get('D_STATE', 8),
                    'hidden_channels': config_dict.get('HIDDEN_CHANNELS', 32)
                }
            # --- FIN ARREGLO ---

            def dict_to_sns(d):
                if isinstance(d, dict):
                    for key, value in d.items():
                        d[key] = dict_to_sns(value)
                    return SimpleNamespace(**d)
                return d

            return dict_to_sns(config_dict)

    except Exception as e:
        logging.error(f"Error al cargar la configuración del experimento '{experiment_name}': {e}", exc_info=True)
        return None


def get_latest_checkpoint(experiment_name: str, silent: bool = False) -> str | None:
    """
    Encuentra la ruta al último checkpoint (.pth) para un experimento dado.

    Args:
        experiment_name: Nombre del experimento
        silent: Si es True, no loguea warnings cuando no hay checkpoints (útil para listar experimentos)
    """
    checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, experiment_name)
    if not os.path.exists(checkpoint_dir):
        # El directorio de checkpoints se crea *dentro* del config.json,
        # así que esto es normal si el experimento es nuevo.
        if not silent:
            logging.info(f"Directorio de checkpoints no encontrado para '{experiment_name}': {checkpoint_dir}")
        return None

    try:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not checkpoints:
            if not silent:
                logging.warning(f"No se encontraron checkpoints .pth en {checkpoint_dir}")
            return None

        # Extraer el número de episodio del nombre del archivo
        def get_episode_num(filename):
            parts = filename.split('_')
            try:
                # Asume formato como '..._ep100.pth' o '..._100.pth'
                num_part = parts[-1].split('.')[0]
                return int(num_part.replace('ep', ''))
            except (ValueError, IndexError):
                return 0  # Si no se puede parsear, se trata como 0

        latest_checkpoint_file = max(checkpoints, key=get_episode_num)
        full_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
        if not silent:
            logging.info(f"Checkpoint más reciente encontrado: {full_path}")
        return full_path

    except Exception as e:
        logging.error(f"Error al buscar el último checkpoint para '{experiment_name}': {e}", exc_info=True)
        return None


def get_latest_jit_model(experiment_name: str, silent: bool = False) -> str | None:
    """
    Encuentra la ruta al último modelo JIT exportado (.pt) para un experimento dado.
    
    Los modelos JIT se guardan en: output/training_checkpoints/<experiment_name>/model_*.pt
    o en: output/jit_models/<experiment_name>/model.pt
    
    Args:
        experiment_name: Nombre del experimento
        silent: Si es True, no loguea warnings cuando no hay modelos
    """
    # Buscar en directorio de checkpoints (mismo lugar que checkpoints)
    checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, experiment_name)
    
    # Buscar modelos JIT
    jit_models = []
    
    # Opción 1: En el directorio de checkpoints
    if os.path.exists(checkpoint_dir):
        jit_models.extend([
            os.path.join(checkpoint_dir, f) 
            for f in os.listdir(checkpoint_dir) 
            if f.endswith('.pt') and (f.startswith('model_') or f == 'model.pt' or f == 'model_jit.pt')
        ])
    
    # Opción 2: En directorio dedicado de modelos JIT
    jit_dir = os.path.join(global_cfg.OUTPUT_DIR, "jit_models", experiment_name)
    if os.path.exists(jit_dir):
        jit_models.extend([
            os.path.join(jit_dir, f) 
            for f in os.listdir(jit_dir) 
            if f.endswith('.pt')
        ])

    if not jit_models:
        if not silent:
            logging.warning(f"No se encontraron modelos JIT (.pt) para '{experiment_name}'")
            logging.info(f"Busca modelos en: {checkpoint_dir} o {jit_dir}")
        return None

    # Retornar el más reciente (por timestamp)
    latest_model = max(jit_models, key=lambda p: os.path.getmtime(p))
    if not silent:
        logging.info(f"Modelo JIT más reciente encontrado: {latest_model}")
    return latest_model


def sns_to_dict_recursive(obj):
    """
    Convierte recursivamente SimpleNamespace (y dicts anidados) a dict.
    """
    if isinstance(obj, SimpleNamespace):
        return {key: sns_to_dict_recursive(value) for key, value in vars(obj).items()}
    elif isinstance(obj, dict):
        return {key: sns_to_dict_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sns_to_dict_recursive(item) for item in obj]
    else:
        # Para tipos primitivos, devolver tal cual
        # Si no es serializable (como torch.device), convertirlo a str
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def check_and_create_dir(exp_config: dict | SimpleNamespace):
    """
    Crea el directorio del experimento y guarda la configuración.
    Convierte SimpleNamespace a dict si es necesario.
    """
    # Convertir recursivamente a dict
    config_dict = sns_to_dict_recursive(exp_config)

    exp_name = config_dict.get("EXPERIMENT_NAME") or config_dict.get("experiment_name")
    if not exp_name:
        logging.error("No se proporcionó EXPERIMENT_NAME en la configuración")
        return

    save_experiment_config(exp_name, config_dict)


def save_experiment_config(experiment_name: str, config_data: dict, is_update: bool = False):
    """
    Guarda el diccionario de configuración en un archivo config.json en el directorio del experimento.

    Args:
        experiment_name: Nombre del experimento
        config_data: Diccionario con la configuración
        is_update: Si es True, actualiza updated_at pero no cambia created_at
    """
    exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, experiment_name)
    config_path = os.path.join(exp_dir, 'config.json')

    os.makedirs(exp_dir, exist_ok=True)

    # Asegurarse de que SimpleNamespace se convierta a dict para JSON
    config_to_save = sns_to_dict_recursive(config_data)

    # Agregar/actualizar timestamps
    now = datetime.now().isoformat()

    if os.path.exists(config_path) and is_update:
        # Si es una actualización, preservar created_at
        try:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
                if 'created_at' in existing_config:
                    config_to_save['created_at'] = existing_config['created_at']
        except:
            pass
        config_to_save['updated_at'] = now
    else:
        # Si es nuevo, establecer ambos timestamps
        if 'created_at' not in config_to_save:
            config_to_save['created_at'] = now
        config_to_save['updated_at'] = now

    try:
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        logging.info(f"Configuración del experimento guardada en: {config_path}")
    except Exception as e:
        logging.error(f"Error al guardar la configuración en '{config_path}': {e}", exc_info=True)

def update_simple_namespace(target: SimpleNamespace, source: SimpleNamespace):
    """
    Actualiza recursivamente un SimpleNamespace con valores de otro.
    """
    for key, value in vars(source).items():
        if isinstance(value, SimpleNamespace) and hasattr(target, key) and isinstance(getattr(target, key), SimpleNamespace):
            update_simple_namespace(getattr(target, key), value)
        else:
            setattr(target, key, value)
