# src/model_loader.py
import torch
import torch.nn as nn
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

def load_checkpoint_data(checkpoint_path: str) -> dict | None:
    """
    Carga los datos de un checkpoint desde un archivo.
    
    Args:
        checkpoint_path: Ruta al archivo de checkpoint.

    Returns:
        Diccionario con los datos del checkpoint o None si falla.
    """
    if not os.path.exists(checkpoint_path):
        logging.error(f"El archivo de checkpoint no se encontró en: {checkpoint_path}")
        return None
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=global_cfg.DEVICE)
        return checkpoint_data
    except Exception as e:
        logging.error(f"Error al cargar el checkpoint '{checkpoint_path}': {e}", exc_info=True)
        return None

def instantiate_model(exp_config: SimpleNamespace) -> nn.Module:
    """
    Instancia un modelo a partir de la configuración del experimento.
    """
    model_name = exp_config.MODEL_ARCHITECTURE
    model_params = exp_config.MODEL_PARAMS
    
    if model_name not in MODEL_MAP:
        raise ValueError(f"Arquitectura de modelo desconocida: '{model_name}'")

    model_class = MODEL_MAP[model_name]
    
    if isinstance(model_params, SimpleNamespace):
        params_dict = vars(model_params)
    else:
        params_dict = model_params

    model = model_class(**params_dict)
    model.to(global_cfg.DEVICE)
    logging.info(f"Modelo '{model_name}' instanciado exitosamente.")
    return model

def load_weights(model: nn.Module, checkpoint_data: dict):
    """
    Carga los pesos desde un diccionario de checkpoint a un modelo.
    Maneja el prefijo '_orig_mod.' de modelos compilados.
    """
    if 'model_state_dict' in checkpoint_data:
        model_state_dict = checkpoint_data['model_state_dict']
    elif 'state_dict' in checkpoint_data:
        model_state_dict = checkpoint_data['state_dict']
    else:
        # Asumir que el diccionario completo es el state_dict
        model_state_dict = checkpoint_data

    # Manejar prefijo de modelo compilado
    if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
        logging.info("Removiendo prefijo '_orig_mod.' del checkpoint.")
        model_state_dict = {
            key.replace('_orig_mod.', '', 1): value
            for key, value in model_state_dict.items()
        }
    
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        logging.warning(f"Claves faltantes al cargar pesos: {missing_keys[:5]}...")
    if unexpected_keys:
        logging.warning(f"Claves inesperadas en los pesos: {unexpected_keys[:5]}...")
        
    logging.info("Pesos del modelo cargados exitosamente.")