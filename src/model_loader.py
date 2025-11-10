# /home/jonathan.correa/Projects/Atheria/src/model_loader.py
import torch
import logging

# Importa las variables de configuración y las arquitecturas de modelo necesarias
from src.config import MODEL_ARCHITECTURE, D_STATE, HIDDEN_CHANNELS
from src.qca_operator_unet_unitary import QCA_Operator_UNet_Unitary
from src.qca_operator_unet import QCA_Operator_UNet
from src.qca_operator_mlp import QCA_Operator_MLP

def load_model():
    """
    Carga y devuelve la arquitectura del modelo especificada en el archivo de configuración.
    """
    logging.info(f"Intentando cargar la arquitectura del modelo: {MODEL_ARCHITECTURE}")
    
    model = None
    
    if MODEL_ARCHITECTURE == "UNET_UNITARIA":
        # Modelo U-Net con pesos unitarios (ideal para estabilidad a largo plazo)
        model = QCA_Operator_UNet_Unitary(d_vector=D_STATE, hidden_channels=HIDDEN_CHANNELS)
        logging.info("Modelo QCA_Operator_UNet_Unitary cargado.")
        
    elif MODEL_ARCHITECTURE == "UNET":
        # Modelo U-Net estándar
        model = QCA_Operator_UNet(d_state=D_STATE, hidden_channels=HIDDEN_CHANNELS)
        logging.info("Modelo QCA_Operator_UNet cargado.")

    elif MODEL_ARCHITECTURE == "MLP":
        # Perceptrón Multicapa simple
        model = QCA_Operator_MLP(d_state=D_STATE, hidden_channels=HIDDEN_CHANNELS)
        logging.info("Modelo QCA_Operator_MLP cargado.")
        
    else:
        # Si la arquitectura no se reconoce, lanza un error claro.
        error_msg = f"Arquitectura de modelo '{MODEL_ARCHITECTURE}' no reconocida en model_loader.py"
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    return model

