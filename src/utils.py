# src/utils.py
import os
import json
import logging
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

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
                    total_training_time = config_data.get('total_training_time', 0)
                    
                    # Si no hay timestamps, usar la fecha de modificación del archivo
                    if not created_at:
                        try:
                            stat = os.stat(config_path)
                            created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
                        except:
                            created_at = datetime.now().isoformat()
                    
                    if not updated_at:
                        try:
                            stat = os.stat(config_path)
                            updated_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
                        except:
                            updated_at = datetime.now().isoformat()
                    
                    # Calcular tiempo total de entrenamiento desde checkpoints si no está guardado
                    if total_training_time == 0 and has_checkpoint:
                        total_training_time = calculate_training_time_from_checkpoints(exp_name)
                    
                    experiments.append({
                        'name': exp_name, 
                        'config': config_data,
                        'has_checkpoint': has_checkpoint,
                        'checkpoint_path': checkpoint_path,
                        'created_at': created_at,
                        'updated_at': updated_at,
                        'last_training_time': last_training_time,
                        'total_training_time': total_training_time
                    })
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logging.error(f"Error al leer o parsear el config.json para el experimento '{exp_name}': {e}. Omitiendo.")
    
    # Ordenar experimentos por fecha de creación (más recientes primero)
    experiments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
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

# --- INICIO DE NUEVAS FUNCIONES (PARA ARREGLAR EL ERROR) ---

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
                return 0 # Si no se puede parsear, se trata como 0
        
        latest_checkpoint_file = max(checkpoints, key=get_episode_num)
        full_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
        logging.info(f"Checkpoint más reciente encontrado: {full_path}")
        return full_path
        
    except Exception as e:
        logging.error(f"Error al buscar el último checkpoint para '{experiment_name}': {e}", exc_info=True)
        return None

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
            import json
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

# --- FIN DE NUEVAS FUNCIONES ---