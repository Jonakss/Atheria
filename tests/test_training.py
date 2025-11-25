# test_training.py
import os
import sys
import logging
from types import SimpleNamespace

# Configurar un logging claro para la prueba
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Añadir el directorio raíz del proyecto al path de Python para que encuentre 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- ¡¡NUEVO!! ARNÉS DE PRUEBAS SECUENCIAL ---
# Se definen configuraciones mínimas para cada modelo para un testeo rápido.
TEST_CONFIGS = [
    {
        "EXPERIMENT_NAME": "TEST_MLP_MINIMAL",
        "MODEL_ARCHITECTURE": "MLP",
        "MODEL_PARAMS": SimpleNamespace(d_state=4, hidden_channels=16),
        "GRID_SIZE_TRAINING": 16,
        "QCA_STEPS_TRAINING": 5,
        "BATCH_SIZE_TRAINING": 2,
        "LR_RATE_M": 1e-4,
        "EPISODES_TO_ADD": 1,
        "STEPS_PER_EPISODE": 2,
        "SAVE_EVERY_EPISODES": 1,
    },
    {
        "EXPERIMENT_NAME": "TEST_UNET_MINIMAL",
        "MODEL_ARCHITECTURE": "UNET",
        "MODEL_PARAMS": SimpleNamespace(d_state=4, hidden_channels=8),
        "GRID_SIZE_TRAINING": 16,
        "QCA_STEPS_TRAINING": 5,
        "BATCH_SIZE_TRAINING": 2,
        "LR_RATE_M": 1e-4,
        "EPISODES_TO_ADD": 1,
        "STEPS_PER_EPISODE": 2,
        "SAVE_EVERY_EPISODES": 1,
    },
    {
        "EXPERIMENT_NAME": "TEST_SNN_UNET_MINIMAL",
        "MODEL_ARCHITECTURE": "SNN_UNET",
        "MODEL_PARAMS": SimpleNamespace(d_state=4, hidden_channels=8, alpha=0.9, beta=0.85),
        "GRID_SIZE_TRAINING": 16,
        "QCA_STEPS_TRAINING": 5,
        "BATCH_SIZE_TRAINING": 2,
        "LR_RATE_M": 1e-4,
        "EPISODES_TO_ADD": 1,
        "STEPS_PER_EPISODE": 2,
        "SAVE_EVERY_EPISODES": 1,
    },
    {
        "EXPERIMENT_NAME": "TEST_DEEP_QCA_MINIMAL",
        "MODEL_ARCHITECTURE": "DEEP_QCA",
        "MODEL_PARAMS": SimpleNamespace(d_state=4, depth=2, hidden_channels=16),
        "GRID_SIZE_TRAINING": 16,
        "QCA_STEPS_TRAINING": 5,
        "BATCH_SIZE_TRAINING": 2,
        "LR_RATE_M": 1e-4,
        "EPISODES_TO_ADD": 1,
        "STEPS_PER_EPISODE": 2,
        "SAVE_EVERY_EPISODES": 1,
    },
    {
        "EXPERIMENT_NAME": "TEST_UNET_UNITARY_MINIMAL",
        "MODEL_ARCHITECTURE": "UNET_UNITARY",
        "MODEL_PARAMS": SimpleNamespace(d_state=4, hidden_channels=8),
        "GRID_SIZE_TRAINING": 16,
        "QCA_STEPS_TRAINING": 5,
        "BATCH_SIZE_TRAINING": 2,
        "LR_RATE_M": 1e-4,
        "EPISODES_TO_ADD": 1,
        "STEPS_PER_EPISODE": 2,
        "SAVE_EVERY_EPISODES": 1,
    },
]

def run_all_tests():
    """
    Ejecuta una prueba de entrenamiento para cada configuración en TEST_CONFIGS.
    """
    # Importar las funciones necesarias una sola vez
    from src.pipeline_train import run_training_pipeline
    from src import config as global_cfg

    total_tests = len(TEST_CONFIGS)
    passed_tests = 0

    for i, test_cfg_dict in enumerate(TEST_CONFIGS):
        model_arch = test_cfg_dict["MODEL_ARCHITECTURE"]
        logging.info(f"\n{'='*80}\n--- INICIANDO PRUEBA {i+1}/{total_tests} PARA EL MODELO: {model_arch} ---\n{'='*80}")
        
        try:
            # Convertir el diccionario a SimpleNamespace para que coincida con la firma de la función
            exp_config = SimpleNamespace(**test_cfg_dict)
            
            # Asegurarse de que los directorios de checkpoints existan para la prueba
            checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_config.EXPERIMENT_NAME)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Llamar directamente a la función principal del entrenamiento
            run_training_pipeline(exp_config)
            
            logging.info(f"--- PRUEBA PARA '{model_arch}' FINALIZADA EXITOSAMENTE ---")
            passed_tests += 1

        except Exception as e:
            logging.error(f"--- LA PRUEBA PARA '{model_arch}' FALLÓ ---", exc_info=True)

    logging.info(f"\n{'='*80}\n--- RESUMEN DE LAS PRUEBAS ---\n{'='*80}")
    logging.info(f"Pruebas completadas: {passed_tests}/{total_tests} pasaron.")
    if passed_tests == total_tests:
        logging.info("¡Felicidades! Todos los modelos son compatibles con el pipeline de entrenamiento.")
    else:
        logging.warning("Algunos modelos fallaron. Revisa los logs de arriba para más detalles.")


if __name__ == "__main__":
    run_all_tests()