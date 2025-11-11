# /home/jonathan.correa/Projects/Atheria/src/model_loader.py
import torch
import logging
import os
import glob

# Importa las variables de configuración y el nuevo sistema de modelos
from . import config as cfg
from . import models

def load_model(model_path=None):
    """
    Carga un modelo y devuelve tanto el modelo como sus metadatos.
    Si se proporciona model_path, carga desde el checkpoint.
    Si no, crea un nuevo modelo basado en la configuración global.
    Devuelve: (model, metadata_dict)
    """
    metadata = {}
    
    # --- Si se proporciona una ruta, cargar desde el checkpoint ---
    if model_path:
        if not os.path.isabs(model_path):
            full_path = os.path.join(cfg.CHECKPOINT_DIR, model_path)
        else:
            full_path = model_path

        if os.path.isdir(full_path):
            logging.info(f"La ruta '{full_path}' es un directorio. Buscando el checkpoint más reciente...")
            checkpoints = glob.glob(os.path.join(full_path, '*.pth'))
            if not checkpoints:
                raise FileNotFoundError(f"No se encontraron archivos .pth en el directorio: {full_path}")
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            full_path = latest_checkpoint

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No se encontró el archivo de checkpoint: {full_path}")

        logging.info(f"Cargando modelo desde el checkpoint: {full_path}")
        checkpoint = torch.load(full_path, map_location=cfg.DEVICE, weights_only=False)

        architecture = checkpoint.get('model_architecture', cfg.MODEL_ARCHITECTURE)
        d_state = checkpoint.get('d_state', cfg.D_STATE)
        hidden_channels = checkpoint.get('hidden_channels', cfg.HIDDEN_CHANNELS)
        
        metadata = {
            'model_architecture': architecture,
            'd_state': d_state,
            'hidden_channels': hidden_channels
        }

        if not all(metadata.values()):
            raise ValueError("El checkpoint o la configuración global no proporcionan los metadatos necesarios.")
        
        if 'model_architecture' not in checkpoint:
            logging.warning(f"Faltan metadatos en el checkpoint. Usando defaults de config: {metadata}")
        else:
            logging.info(f"Metadatos del checkpoint: {metadata}")

        ModelClass = models.get_model_class(architecture)
        
        if architecture == "UNET_UNITARIA":
            model = ModelClass(d_vector=d_state, hidden_channels=hidden_channels)
        elif architecture == "SNN_UNET":
            model = ModelClass(in_channels=d_state, out_channels=d_state, hidden_channels=hidden_channels)
        else:
            model = ModelClass(d_state=d_state, hidden_channels=hidden_channels)
            
        state_dict = checkpoint['model_state_dict']
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        load_result = model.load_state_dict(state_dict, strict=False)
        
        # (Logging de claves faltantes/inesperadas)
        if load_result.missing_keys or load_result.unexpected_keys:
            logging.warning(f"Claves faltantes: {load_result.missing_keys}")
            logging.warning(f"Claves inesperadas: {load_result.unexpected_keys}")

        logging.info("Modelo cargado y pesos aplicados desde el checkpoint.")

    # --- Si no se proporciona ruta, crear un modelo nuevo desde config ---
    else:
        logging.info(f"Creando un nuevo modelo desde la configuración global: {cfg.MODEL_ARCHITECTURE}")
        
        metadata = {
            'model_architecture': cfg.MODEL_ARCHITECTURE,
            'd_state': cfg.D_STATE,
            'hidden_channels': cfg.HIDDEN_CHANNELS
        }
        
        ModelClass = models.get_model_class(cfg.MODEL_ARCHITECTURE)
        
        if cfg.MODEL_ARCHITECTURE == "UNET_UNITARIA":
            model = ModelClass(d_vector=cfg.D_STATE, hidden_channels=cfg.HIDDEN_CHANNELS)
        elif cfg.MODEL_ARCHITECTURE == "SNN_UNET":
            model = ModelClass(in_channels=cfg.D_STATE, out_channels=cfg.D_STATE, hidden_channels=cfg.HIDDEN_CHANNELS)
        else:
            model = ModelClass(d_state=cfg.D_STATE, hidden_channels=cfg.HIDDEN_CHANNELS)

        logging.info("Nuevo modelo creado exitosamente.")

    return model, metadata

