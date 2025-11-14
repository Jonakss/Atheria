# src/models/__init__.py
"""
Este paquete contiene las arquitecturas de los modelos de Ley M.

Cada fichero define una clase de `torch.nn.Module`. Este __init__.py
importa explícitamente cada clase para que puedan ser fácilmente
referenciadas por el resto de la aplicación, como en el MODEL_MAP
del model_loader.
"""
from .unet import UNet
from .unet_unitary import UNetUnitary
from .snn_unet import SNNUNet
from .mlp import MLP
from .deep_qca import DeepQCA
