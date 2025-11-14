# src/utils.py
import os
import json
import logging
import glob
from types import SimpleNamespace
from . import config as cfg

def get_experiment_list():
    """
    Escanea el directorio de experimentos de forma robusta.
    Devuelve una lista de experimentos que tienen un config.json válido.
    """
    exp_list = []
    if not os.path.exists(cfg.EXPERIMENTS_DIR):
        os.makedirs(cfg.EXPERIMENTS_DIR)
        return exp_list

    for exp_name in sorted(os.listdir(cfg.EXPERIMENTS_DIR)):
        exp_path = os.path.join(cfg.EXPERIMENTS_DIR, exp_name)
        if os.path.isdir(exp_path):
            config_path = os.path.join(exp_path, 'config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    # Verificación mínima: el config debe ser un diccionario
                    if isinstance(config_data, dict):
                        exp_list.append({'name': exp_name, 'config': config_data})
                    else:
                        logging.warning(f"El config.json de '{exp_name}' no es un objeto JSON válido. Omitiendo.")
                except Exception as e:
                    logging.error(f"Error al leer o parsear el config.json para el experimento '{exp_name}': {e}. Omitiendo.")
            # No añadir experimentos sin config.json a la lista
    return exp_list

def load_experiment_config(experiment_name):
    """Carga la configuración de un experimento y la devuelve como un SimpleNamespace."""
    config_path = os.path.join(cfg.EXPERIMENTS_DIR, experiment_name, 'config.json')
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return SimpleNamespace(**config_dict)
    except FileNotFoundError:
        logging.error(f"No se encontró el fichero de configuración para el experimento: {experiment_name}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error al decodificar el JSON del fichero de configuración para el experimento: {experiment_name}")
        return None

def save_experiment_config(experiment_name, config_dict):
    """Guarda la configuración de un experimento."""
    exp_dir = os.path.join(cfg.EXPERIMENTS_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def get_latest_checkpoint(experiment_name, checkpoint_type="qca"):
    """Encuentra el último checkpoint para un tipo dado en un experimento."""
    checkpoint_dir = os.path.join(cfg.EXPERIMENTS_DIR, experiment_name, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    
    search_pattern = os.path.join(checkpoint_dir, f"{checkpoint_type}_checkpoint_eps*.pth")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        # Si no hay checkpoints de episodios, buscar el 'best'
        best_file = os.path.join(checkpoint_dir, f"{checkpoint_type}_best.pth")
        if os.path.exists(best_file):
            return best_file
        return None
        
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
