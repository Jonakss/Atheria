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
from .models.unet_convlstm import UNetConvLSTM
from .models.unet_unitary_rmsnorm import UNetUnitaryRMSNorm
from .utils import get_latest_checkpoint # Importar la utilidad

MODEL_MAP = {
    "UNET": UNet,
    "SNN_UNET": SNNUNet,
    "DEEP_QCA": DeepQCA,
    "MLP": MLP,
    "UNET_UNITARY": UNetUnitary,
    "UNET_CONVLSTM": UNetConvLSTM,
    "UNET_UNITARY_RMSNORM": UNetUnitaryRMSNorm,
}

def create_new_model(exp_config: dict | SimpleNamespace):
    """
    Crea un nuevo modelo sin cargar pesos entrenados.
    Útil para entrenamiento desde cero.
    
    Args:
        exp_config: Configuración del experimento (dict o SimpleNamespace)
    
    Returns:
        Modelo instanciado (sin pesos entrenados)
    """
    # Convertir dict a SimpleNamespace si es necesario
    if isinstance(exp_config, dict):
        exp_cfg = SimpleNamespace(**exp_config)
        exp_cfg.MODEL_PARAMS = SimpleNamespace(**exp_config.get("MODEL_PARAMS", {}))
    else:
        exp_cfg = exp_config
    
    model_name = exp_cfg.MODEL_ARCHITECTURE
    model_params = exp_cfg.MODEL_PARAMS
    
    if model_name not in MODEL_MAP:
        logging.error(f"Arquitectura de modelo desconocida: '{model_name}'")
        raise ValueError(f"Arquitectura de modelo desconocida: '{model_name}'")

    # Instanciar la arquitectura del modelo
    model_class = MODEL_MAP[model_name]
    
    # Convertir model_params a dict si es SimpleNamespace
    if isinstance(model_params, SimpleNamespace):
        params_dict = vars(model_params)
    else:
        params_dict = model_params
    
    model = model_class(**params_dict)
    model.to(global_cfg.DEVICE)
    model.train()  # Modo entrenamiento
    
    logging.info(f"Modelo '{model_name}' creado exitosamente para entrenamiento.")
    return model

def load_model(exp_cfg: SimpleNamespace, checkpoint_path: str | None = None):
    """
    Carga la arquitectura de un modelo.
    Si se proporciona un checkpoint_path, carga los pesos entrenados.
    Devuelve (model, state_dict) o (model, None).
    """
    model_name = exp_cfg.MODEL_ARCHITECTURE
    model_params = exp_cfg.MODEL_PARAMS
    
    if model_name not in MODEL_MAP:
        logging.error(f"Arquitectura de modelo desconocida: '{model_name}'")
        raise ValueError(f"Arquitectura de modelo desconocida: '{model_name}'")

    # 1. Instanciar la arquitectura del modelo
    model_class = MODEL_MAP[model_name]
    
    # Convertir model_params a dict si es SimpleNamespace
    if isinstance(model_params, SimpleNamespace):
        params_dict = vars(model_params)
    else:
        params_dict = model_params
    
    model = model_class(**params_dict)
    logging.info(f"Modelo '{model_name}' instanciado exitosamente.")

    # 2. Cargar los pesos si se proporciona un path
    state_dict = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=global_cfg.DEVICE)
            
            # Manejar diferentes formatos de checkpoint
            # Formato nuevo: {'model_state_dict': {...}, 'episode': ..., ...}
            # Formato antiguo: directamente el state_dict
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    model_state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    model_state_dict = state_dict['state_dict']
                else:
                    # Si no tiene las claves esperadas, asumir que el dict completo es el state_dict
                    model_state_dict = state_dict
            else:
                # Si no es un dict, asumir que es directamente el state_dict
                model_state_dict = state_dict
                state_dict = {'model_state_dict': model_state_dict}
            
            # Manejar checkpoints de modelos compilados (torch.compile) que tienen prefijo "_orig_mod."
            # Remover el prefijo si existe
            if isinstance(model_state_dict, dict) and any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
                logging.info("Checkpoint contiene modelo compilado, removiendo prefijo '_orig_mod.'")
                cleaned_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith('_orig_mod.'):
                        new_key = key.replace('_orig_mod.', '', 1)
                        cleaned_state_dict[new_key] = value
                    else:
                        cleaned_state_dict[key] = value
                model_state_dict = cleaned_state_dict
            
            # Cargar los pesos
            if isinstance(model_state_dict, dict):
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                if missing_keys:
                    logging.warning(f"Claves faltantes al cargar checkpoint: {missing_keys[:5]}... (mostrando primeras 5)")
                if unexpected_keys:
                    logging.warning(f"Claves inesperadas en checkpoint: {unexpected_keys[:5]}... (mostrando primeras 5)")
            else:
                logging.error(f"Formato de checkpoint inesperado. model_state_dict no es un dict: {type(model_state_dict)}")
                return None, None
            
            model.to(global_cfg.DEVICE)
            model.eval() # Poner el modelo en modo de evaluación
            logging.info(f"Pesos del modelo cargados desde el checkpoint: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error al cargar el checkpoint '{checkpoint_path}': {e}. Se usará un modelo sin entrenar.", exc_info=True)
            # Anular si la carga falla y devolver None para que el servidor lo detecte
            state_dict = None
            return None, None
    else:
        logging.warning(f"No se proporcionó un checkpoint o no se encontró. El modelo '{model_name}' se usará sin entrenar.")
        model.to(global_cfg.DEVICE)
        
    return model, state_dict