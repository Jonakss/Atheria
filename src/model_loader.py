# /home/jonathan.correa/Projects/Atheria/src/model_loader.py
import torch
import logging
import os

# Importa las variables de configuración y el nuevo sistema de modelos
from . import config as cfg
from . import models

def load_model(model_path=None):
    """
    Carga un modelo. Si se proporciona model_path, carga desde el checkpoint.
    Si no, crea un nuevo modelo basado en la configuración global.
    """
    
    # --- Si se proporciona una ruta, cargar desde el checkpoint ---
    if model_path:
        # Construir la ruta completa si es relativa
        if not os.path.isabs(model_path):
            full_path = os.path.join(cfg.CHECKPOINT_DIR, model_path)
        else:
            full_path = model_path

        if not os.path.exists(full_path):
            error_msg = f"No se encontró el archivo de checkpoint: {full_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        logging.info(f"Cargando modelo desde el checkpoint: {full_path}")
        checkpoint = torch.load(full_path, map_location=cfg.DEVICE)

        # Extraer metadatos del checkpoint
        architecture = checkpoint.get('model_architecture')
        d_state = checkpoint.get('d_state')
        hidden_channels = checkpoint.get('hidden_channels')

        if not all([architecture, d_state, hidden_channels]):
            error_msg = "El checkpoint no contiene los metadatos necesarios (model_architecture, d_state, hidden_channels)."
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        logging.info(f"Metadatos del checkpoint: Arch={architecture}, D_State={d_state}, HiddenCh={hidden_channels}")

        # Instanciar el modelo correcto usando el sistema de registro
        ModelClass = models.get_model_class(architecture)
        
        # Manejar el caso especial del constructor de UNET_UNITARIA
        if architecture == "UNET_UNITARIA":
            model = ModelClass(d_vector=d_state, hidden_channels=hidden_channels)
        else:
            model = ModelClass(d_state=d_state, hidden_channels=hidden_channels)
            
        # Cargar los pesos (state_dict)
        state_dict = checkpoint['model_state_dict']
        # Manejar el caso en que el modelo se guardó con DataParallel (contiene 'module.')
        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        
        logging.info("Modelo cargado y pesos aplicados desde el checkpoint exitosamente.")

    # --- Si no se proporciona ruta, crear un modelo nuevo desde config ---
    else:
        logging.info(f"No se proporcionó model_path. Creando un nuevo modelo desde la configuración global: {cfg.MODEL_ARCHITECTURE}")
        
        ModelClass = models.get_model_class(cfg.MODEL_ARCHITECTURE)
        
        # Manejar el caso especial del constructor de UNET_UNITARIA
        if cfg.MODEL_ARCHITECTURE == "UNET_UNITARIA":
            model = ModelClass(d_vector=cfg.D_STATE, hidden_channels=cfg.HIDDEN_CHANNELS)
        else:
            model = ModelClass(d_state=cfg.D_STATE, hidden_channels=cfg.HIDDEN_CHANNELS)

        logging.info("Nuevo modelo creado exitosamente.")

    return model

