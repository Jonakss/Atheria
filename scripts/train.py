# train.py
import os
import sys
import asyncio
import argparse # <-- ¡NUEVO! Para leer argumentos de la terminal

# --- Configuración del Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Importa solo los módulos que necesitas para entrenar
try:
    from src.pipeline_train import run_training_pipeline
    from src import config as cfg # Moved import here
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos desde 'src': {e}", file=sys.stderr)
    sys.exit(1)

def main_training(args, cfg_module):
    """
    Punto de entrada dedicado solo para el entrenamiento.
    Ahora usa 'args' en lugar de 'cfg_module' para los parámetros clave.
    """
    print("================================================")
    print("🔬 INICIANDO TRABAJO DE ENTRENAMIENTO (MODO WORKER) 🔬")
    print("================================================")
    
    # --- ¡NUEVO! Sobrescribir el config con los argumentos ---
    # Esto permite que el "Laboratorio" controle la configuración
    cfg_module.EXPERIMENT_NAME = args.name
    cfg_module.HIDDEN_CHANNELS = args.hidden_channels
    cfg_module.LR_RATE_M = args.lr
    cfg_module.EPISODES_TO_ADD = args.episodes
    cfg_module.GRID_SIZE_TRAINING = args.training_grid_size # NEW: Set training grid size
    
    # Seleccionar el modelo
    if args.model == 'unet':
        cfg_module.ACTIVE_QCA_OPERATOR = 'UNET'
    elif args.model == 'unet_unitary':
        cfg_module.ACTIVE_QCA_OPERATOR = 'UNET_UNITARIA'
    else:
        cfg_module.ACTIVE_QCA_OPERATOR = 'MLP'
    # -------------------------------------------------

    # --- Imprimir configuración para logging ---
    print("Configuración de Entrenamiento:")
    print(f"DEBUG: type(cfg_module) = {type(cfg_module)}")
    print(f"  - Nombre Experimento: {cfg_module.EXPERIMENT_NAME}")
    print(f"  - Modelo:             {cfg_module.ACTIVE_QCA_OPERATOR}")
    print(f"  - Canales Ocultos:    {cfg_module.HIDDEN_CHANNELS}")
    print(f"  - Tasa de Aprendizaje:  {cfg_module.LR_RATE_M}")
    print(f"  - Episodios:          {cfg_module.EPISODES_TO_ADD}")
    print(f"  - Tamaño Grilla Entrenamiento: {cfg_module.GRID_SIZE_TRAINING}") # NEW: Log training grid size
    print(f"  - Dimensión Estado (D): {cfg_module.D_STATE}")
    print("-------------------------------------------------")

    try:
        # Llamar a la función de pipeline (que ahora leerá el cfg actualizado)
        run_training_pipeline(cfg_module) # Pass cfg_module explicitly
        print("\n✅ Entrenamiento finalizado con éxito.")
    except Exception as e:
        print(f"\n❌ El entrenamiento falló con un error crítico: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento interrumpido por el usuario.")

# ==============================================================================
# PUNTO DE ENTRADA DEL SCRIPT DE ENTRENAMIENTO
# ==============================================================================
if __name__ == "__main__":
    print("DEBUG: train.py script started.") # Added debug print
    from src import config as cfg # Moved import here
    # --- ¡NUEVO! Definir los argumentos de línea de comandos ---
    parser = argparse.ArgumentParser(description="Lanzador de Entrenamiento AETHERIA")
    parser.add_argument('--name', type=str, default="DefaultExperiment", help='Nombre de la carpeta del experimento')
    parser.add_argument('--model', type=str, default='unet', choices=['mlp', 'unet', 'unet_unitary'], help='Modelo a entrenar')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Canales ocultos (ancho) del modelo')
    parser.add_argument('--lr', type=float, default=1e-6, help='Tasa de aprendizaje')
    parser.add_argument('--episodes', type=int, default=2000, help='Número de episodios a entrenar')
    parser.add_argument('--training_grid_size', type=int, default=64, help='Tamaño de la grilla para entrenamiento') # NEW
    
    args = parser.parse_args()
    print(f"DEBUG: Arguments received by train.py: {args}") # Added debug print
    
    # Override with cfg defaults if not provided by command line
    if args.name == "DefaultExperiment":
        args.name = cfg.EXPERIMENT_NAME
    if args.hidden_channels == 32: # Assuming 32 is the default in cfg
        args.hidden_channels = cfg.HIDDEN_CHANNELS
    if args.lr == 1e-6: # Assuming 1e-6 is the default in cfg
        args.lr = cfg.LR_RATE_M
    if args.episodes == 2000: # Assuming 2000 is the default in cfg
        args.episodes = cfg.EPISODES_TO_ADD
    if args.training_grid_size == 64: # Assuming 64 is the default in cfg
        args.training_grid_size = cfg.GRID_SIZE_TRAINING
    # Model type default is 'unet', which is handled by the if/elif in main_training
    
    main_training(args, cfg)