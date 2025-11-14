# src/model_loader.py
import torch
import logging
from types import SimpleNamespace

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

def load_model(experiment_config: SimpleNamespace, checkpoint_path: str = None):
    """
    Carga un modelo de forma robusta basado en la configuración del experimento.
    """
    try:
        model_arch_name = experiment_config.MODEL_ARCHITECTURE
        model_class = MODEL_MAP.get(model_arch_name)

        if not model_class:
            logging.error(f"Arquitectura de modelo desconocida: '{model_arch_name}'")
            return None

        # --- ¡¡SOLUCIÓN DEFINITIVA!! Convertir SimpleNamespace a dict de forma robusta ---
        model_params_ns = experiment_config.MODEL_PARAMS
        model_params_dict = _namespace_to_dict(model_params_ns)
        
        model = model_class(**model_params_dict)
        logging.info(f"Modelo '{model_arch_name}' instanciado exitosamente.")

        if checkpoint_path:
            logging.info(f"Cargando checkpoint desde: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            is_compiled = any(k.startswith('_orig_mod.') for k in model.state_dict())
            
            if is_compiled and not all(k.startswith('_orig_mod.') for k in state_dict):
                logging.info("Adaptando checkpoint para modelo compilado...")
                state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            logging.info("Checkpoint cargado exitosamente.")

        return model

    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}", exc_info=True)
        return None