# scripts/train.py
import os
import sys
import json
import argparse
import logging

# --- ¡¡VERIFICACIÓN!! Logear tan pronto como sea posible ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("El script train.py ha comenzado a ejecutarse.")

# Añadir el directorio raíz del proyecto al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.info(f"Python path actualizado: {sys.path}")

def main():
    parser = argparse.ArgumentParser(description="Lanzador de entrenamiento para un experimento de Atheria.")
    parser.add_argument('--experiment_name', type=str, required=True, help='El nombre del experimento a entrenar.')
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    
    print(f"--- INICIANDO ENTRENAMIENTO PARA EL EXPERIMENTO: {experiment_name} ---", flush=True)

    try:
        logging.info("Importando módulos del proyecto...")
        from src.utils import load_experiment_config
        from src.pipelines.pipeline_train import run_training_pipeline
        logging.info("Módulos importados exitosamente.")
        
        logging.info(f"Cargando configuración para el experimento: {experiment_name}")
        config = load_experiment_config(experiment_name)
        if not config:
            raise FileNotFoundError(f"No se pudo cargar la configuración para el experimento '{experiment_name}'.")
        logging.info("Configuración cargada exitosamente.")
        
        logging.info("Iniciando el pipeline de entrenamiento...")
        run_training_pipeline(config)
        logging.info("El pipeline de entrenamiento ha finalizado.")

    except Exception as e:
        logging.error(f"El entrenamiento de '{experiment_name}' falló: {e}", exc_info=True)
        # El traceback se imprimirá gracias a exc_info=True
        sys.exit(1)

if __name__ == "__main__":
    main()
