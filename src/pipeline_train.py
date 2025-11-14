# src/pipeline_train.py
import torch
from types import SimpleNamespace
import logging

from . import config as global_cfg
from .model_loader import load_model
from .qca_engine import Aetheria_Motor
from .trainer import QC_Trainer_v3

def run_training_pipeline(exp_cfg: SimpleNamespace):
    """
    Ejecuta el pipeline de entrenamiento completo para una configuración de experimento dada.
    """
    # --- ¡¡CORRECCIÓN!! exp_cfg ya es un SimpleNamespace, no se necesita conversión ---
    
    device = global_cfg.get_device()
    logging.info(f"Pipeline de entrenamiento iniciado en el dispositivo: {device}")

    # 1. Cargar Ley M (el modelo)
    ley_M = load_model(exp_cfg)
    if ley_M is None:
        logging.error("No se pudo cargar el modelo. Abortando entrenamiento.")
        return

    # 2. Inicializar el Motor Aetheria
    motor = Aetheria_Motor(ley_M, exp_cfg.GRID_SIZE_TRAINING, exp_cfg.MODEL_PARAMS.d_state, device)

    # 3. Compilar el modelo si es compatible
    if getattr(ley_M, '_compiles', True):
        motor.compile_model()
    else:
        logging.info(f"El modelo '{exp_cfg.MODEL_ARCHITECTURE}' ha deshabilitado torch.compile(). Se ejecutará sin compilar.")
    
    num_params = sum(p.numel() for p in motor.operator.parameters() if p.requires_grad)
    print(f"Motor y Ley-M ({exp_cfg.MODEL_ARCHITECTURE}) inicializados. Cuadrícula: {exp_cfg.GRID_SIZE_TRAINING}x{exp_cfg.GRID_SIZE_TRAINING}.")
    print(f"Parámetros Entrenables: {num_params}")

    # 4. Inicializar el Entrenador
    trainer = QC_Trainer_v3(motor, exp_cfg.LR_RATE_M, global_cfg, exp_cfg)

    # 5. Bucle de Entrenamiento (ahora gestionado por el trainer)
    trainer.run_training_loop()

    logging.info("Pipeline de entrenamiento finalizado.")
