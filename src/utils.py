# src/utils.py
import os
import json
import logging
from types import SimpleNamespace

from . import config as global_cfg

def get_experiment_list():
    """
    Escanea el directorio de experimentos y devuelve una lista de experimentos válidos.
    Ahora es robusto a ficheros config.json corruptos.
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
                    experiments.append({'name': exp_name, 'config': config_data})
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logging.error(f"Error al leer o parsear el config.json para el experimento '{exp_name}': {e}. Omitiendo.")
    return experiments

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
            
            # --- ¡¡SOLUCIÓN DEFINITIVA!! Convertir recursivamente a SimpleNamespace ---
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