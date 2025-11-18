# src/pipeline_train.py
import torch
from types import SimpleNamespace
import logging

from . import config as global_cfg
from .model_loader import create_new_model, load_model
from .qca_engine import Aetheria_Motor
from .qc_trainer import QC_Trainer_v3
from .utils import get_latest_checkpoint

def run_training_pipeline(exp_cfg: SimpleNamespace, checkpoint_path: str | None = None):
    """
    Ejecuta el pipeline de entrenamiento completo para una configuración de experimento dada.
    
    Args:
        exp_cfg: Configuración del experimento (SimpleNamespace)
        checkpoint_path: Ruta al checkpoint si se está continuando entrenamiento (opcional)
    """
    device = global_cfg.get_device()
    logging.info(f"Pipeline de entrenamiento iniciado en el dispositivo: {device}")

    # 1. Cargar o crear el modelo
    if checkpoint_path:
        # Continuar entrenamiento: cargar modelo con checkpoint del mismo experimento
        ley_M, state_dict = load_model(exp_cfg, checkpoint_path)
        if ley_M is None:
            logging.error("No se pudo cargar el modelo desde el checkpoint. Abortando entrenamiento.")
            return
        # El modelo ya está en modo eval después de load_model, cambiar a train
        ley_M.train()
        logging.info(f"Modelo cargado desde checkpoint: {checkpoint_path}")
    elif hasattr(exp_cfg, 'LOAD_FROM_EXPERIMENT') and exp_cfg.LOAD_FROM_EXPERIMENT:
        # Transfer Learning: crear nuevo modelo y cargar pesos de otro experimento
        try:
            ley_M = create_new_model(exp_cfg)
            logging.info(f"Nuevo modelo creado. Cargando pesos desde '{exp_cfg.LOAD_FROM_EXPERIMENT}' para transfer learning...")
            
            # Cargar checkpoint del experimento base
            from .utils import get_latest_checkpoint
            base_checkpoint = get_latest_checkpoint(exp_cfg.LOAD_FROM_EXPERIMENT)
            if base_checkpoint:
                try:
                    checkpoint = torch.load(base_checkpoint, map_location=device)
                    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
                    
                    # Manejar prefijo _orig_mod. si existe
                    if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
                        cleaned_state_dict = {}
                        for key, value in model_state_dict.items():
                            if key.startswith('_orig_mod.'):
                                cleaned_state_dict[key.replace('_orig_mod.', '', 1)] = value
                            else:
                                cleaned_state_dict[key] = value
                        model_state_dict = cleaned_state_dict
                    
                    # Cargar con strict=False para permitir diferencias de arquitectura
                    ley_M.load_state_dict(model_state_dict, strict=False)
                    logging.info(f"✅ Transfer learning: pesos cargados desde '{exp_cfg.LOAD_FROM_EXPERIMENT}'. Iniciando desde episodio 0.")
                except Exception as e:
                    logging.warning(f"⚠️ No se pudieron cargar pesos desde '{exp_cfg.LOAD_FROM_EXPERIMENT}': {e}. Iniciando desde cero.")
            else:
                logging.warning(f"⚠️ No se encontró checkpoint para '{exp_cfg.LOAD_FROM_EXPERIMENT}'. Iniciando desde cero.")
            
            ley_M.train()
        except Exception as e:
            logging.error(f"Error al crear el modelo: {e}", exc_info=True)
            return
    else:
        # Entrenamiento desde cero: crear nuevo modelo
        try:
            ley_M = create_new_model(exp_cfg)
            logging.info("Nuevo modelo creado para entrenamiento desde cero.")
            ley_M.train()
        except Exception as e:
            logging.error(f"Error al crear el modelo: {e}", exc_info=True)
            return

    # 2. Inicializar el Motor Aetheria
    # Pasar exp_cfg para que el motor tenga acceso a GAMMA_DECAY (término Lindbladian)
    motor = Aetheria_Motor(ley_M, exp_cfg.GRID_SIZE_TRAINING, exp_cfg.MODEL_PARAMS.d_state, device, cfg=exp_cfg)

    # 3. Compilar el modelo si es compatible
    if getattr(ley_M, '_compiles', True):
        motor.compile_model()
    else:
        logging.info(f"El modelo '{exp_cfg.MODEL_ARCHITECTURE}' ha deshabilitado torch.compile(). Se ejecutará sin compilar.")
    
    # Usar get_model_for_params() para acceder a los parámetros correctamente
    # (maneja el caso cuando el modelo está compilado)
    model_for_params = motor.get_model_for_params()
    num_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
    print(f"Motor y Ley-M ({exp_cfg.MODEL_ARCHITECTURE}) inicializados. Cuadrícula: {exp_cfg.GRID_SIZE_TRAINING}x{exp_cfg.GRID_SIZE_TRAINING}.")
    print(f"Parámetros Entrenables: {num_params}")

    # 4. Inicializar el Entrenador
    trainer = QC_Trainer_v3(motor, exp_cfg.LR_RATE_M, global_cfg, exp_cfg)

    # 5. Bucle de Entrenamiento (ahora gestionado por el trainer)
    trainer.run_training_loop()

    logging.info("Pipeline de entrenamiento finalizado.")
