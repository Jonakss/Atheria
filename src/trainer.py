# src/trainer.py
import argparse
import json
import logging
from types import SimpleNamespace

from . import config as global_cfg
from .utils import check_and_create_dir, load_experiment_config, get_latest_checkpoint
from .pipelines.pipeline_train import run_training_pipeline

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sns_to_dict(obj):
    """
    Convierte recursivamente SimpleNamespace (y dicts anidados) a dict para serialización JSON.
    """
    if isinstance(obj, SimpleNamespace):
        return {key: sns_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, dict):
        return {key: sns_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sns_to_dict(item) for item in obj]
    else:
        # Para tipos primitivos (int, float, str, bool, None) y objetos no serializables
        # Intentar serializar, si falla devolver str
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

def main():
    parser = argparse.ArgumentParser(description="Aetheria Experiment Trainer")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--model_architecture", type=str, required=True, choices=["MLP", "UNET", "UNET_COSMOLOGICA", "UNET_UNITARIA", "UNET_UNITARY", "UNET_UNITARY_RMSNORM", "UNET_CONVLSTM", "SNN_UNET", "DEEP_QCA"])
    parser.add_argument("--lr_rate_m", type=float, required=True)
    parser.add_argument("--grid_size_training", type=int, required=True)
    parser.add_argument("--qca_steps_training", type=int, required=True)
    parser.add_argument("--total_episodes", type=int, required=True)
    # --- LA CORRECCIÓN CLAVE ESTÁ AQUÍ ---
    # Aceptamos un string JSON y lo cargamos como un diccionario
    parser.add_argument("--model_params", type=str, required=True, help='JSON string of model parameters')
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--noise_level", type=float, default=0.05, help="Maximum noise level for training")
    
    args = parser.parse_args()
    
    # Decodificamos el string JSON a un diccionario de Python
    try:
        model_params_dict = json.loads(args.model_params)
    except json.JSONDecodeError as e:
        print(f"Error: No se pudo decodificar --model_params. Asegúrate de que es un JSON válido. Error: {e}")
        return

    # Construimos el objeto de configuración del experimento
    # El model_loader ya maneja los nombres correctamente, así que usamos el valor tal cual
    exp_config = {
        "EXPERIMENT_NAME": args.experiment_name,
        "MODEL_ARCHITECTURE": args.model_architecture,
        "LR_RATE_M": args.lr_rate_m,
        "GRID_SIZE_TRAINING": args.grid_size_training,
        "QCA_STEPS_TRAINING": args.qca_steps_training,
        "TOTAL_EPISODES": args.total_episodes,
        "MODEL_PARAMS": model_params_dict,
        "DEVICE": global_cfg.DEVICE,
        "GAMMA_DECAY": getattr(global_cfg, 'GAMMA_DECAY', 0.01),  # Término Lindbladian (decaimiento)
        "NOISE_LEVEL": args.noise_level
    }
    
    # Convertir MODEL_PARAMS a SimpleNamespace si es necesario para compatibilidad
    if isinstance(exp_config["MODEL_PARAMS"], dict):
        exp_config["MODEL_PARAMS"] = SimpleNamespace(**exp_config["MODEL_PARAMS"])

    # Agregar CONTINUE_TRAINING a la config
    exp_config["CONTINUE_TRAINING"] = args.continue_training
    
    # Convertir toda la config a SimpleNamespace
    exp_cfg = SimpleNamespace(**exp_config)

    # Mostrar configuración (convertir a dict para JSON)
    config_for_display = sns_to_dict(exp_config)
    print(f"Iniciando entrenamiento con la siguiente configuración:\n{json.dumps(config_for_display, indent=2)}")

    # check_and_create_dir acepta dict o SimpleNamespace, pero usa el dict original
    check_and_create_dir(exp_config)
    
    # Si continuamos entrenamiento, cargar checkpoint
    checkpoint_path = None
    if args.continue_training:
        checkpoint_path = get_latest_checkpoint(exp_config["EXPERIMENT_NAME"])
        if checkpoint_path:
            print(f"Continuando entrenamiento desde: {checkpoint_path}")
            # Actualizar el episodio inicial si hay checkpoint
            # START_EPISODE será el episodio del checkpoint + 1 (para continuar desde el siguiente)
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                checkpoint_episode = checkpoint.get('episode', 0)
                exp_cfg.START_EPISODE = checkpoint_episode + 1
                logging.info(f"Checkpoint encontrado en episodio {checkpoint_episode}, continuando desde {exp_cfg.START_EPISODE}")
            except Exception as e:
                logging.warning(f"No se pudo leer el episodio del checkpoint: {e}")
                exp_cfg.START_EPISODE = 0
        else:
            print("No se encontró checkpoint, iniciando desde cero.")
            exp_cfg.START_EPISODE = 0
    else:
        exp_cfg.START_EPISODE = 0
    
    # Ejecutar el pipeline de entrenamiento
    try:
        run_training_pipeline(exp_cfg, checkpoint_path=checkpoint_path)
    except Exception as e:
        logging.error(f"Error en el pipeline de entrenamiento: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()