# src/model_loader.py
import torch
import logging
from types import SimpleNamespace
import os

from . import config as global_cfg
from . import models
from .models.unet import UNet
from .models.snn_unet import SNNUNet
from .models.deep_qca import DeepQCA
from .models.mlp import MLP
from .models.unet_unitary import UNetUnitary

MODEL_MAP = {
    "UNET": UNet,
    "SNN_UNET": SNNUNet,
    "DEEP_QCA": DeepQCA,
    "MLP": MLP,
    "UNET_UNITARY": UNetUnitary,
}

def _namespace_to_dict(ns):
    """Convierte recursivamente un SimpleNamespace a un diccionario."""
    if not isinstance(ns, SimpleNamespace):
        return ns
    return {key: _namespace_to_dict(value) for key, value in ns.__dict__.items()}

def load_model(exp_cfg: SimpleNamespace):
    """
    Carga la arquitectura de un modelo y los pesos entrenados desde un checkpoint.
    """
    model_name = exp_cfg.MODEL_ARCHITECTURE
    model_params = exp_cfg.MODEL_PARAMS
    
    if model_name not in MODEL_MAP:
        logging.error(f"Arquitectura de modelo desconocida: '{model_name}'")
        raise ValueError(f"Arquitectura de modelo desconocida: '{model_name}'")

    # 1. Instanciar la arquitectura del modelo
    model_class = MODEL_MAP[model_name]
    model = model_class(**vars(model_params))
    logging.info(f"Modelo '{model_name}' instanciado exitosamente.")

    # 2. --- ¡¡CORRECCIÓN CLAVE!! Cargar los pesos del último checkpoint ---
    checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_cfg.EXPERIMENT_NAME)
    if os.path.exists(checkpoint_dir):
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                
                state_dict = torch.load(checkpoint_path, map_location=global_cfg.DEVICE)
                model.load_state_dict(state_dict['model_state_dict'])
                model.to(global_cfg.DEVICE)
                model.eval() # Poner el modelo en modo de evaluación
                logging.info(f"Pesos del modelo cargados desde el checkpoint: {latest_checkpoint}")
            else:
                logging.warning(f"No se encontraron checkpoints en {checkpoint_dir}, se usará un modelo sin entrenar.")
        except Exception as e:
            logging.error(f"Error al cargar el checkpoint para '{exp_cfg.EXPERIMENT_NAME}': {e}. Se usará un modelo sin entrenar.", exc_info=True)
    else:
        logging.warning(f"El directorio de checkpoints no existe: {checkpoint_dir}. Se usará un modelo sin entrenar.")
        
    return model