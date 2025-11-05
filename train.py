# train.py
import os
import sys
import asyncio
import argparse # <-- ¡NUEVO! Para leer argumentos de la terminal

# --- Configuración del Path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Importa solo los módulos que necesitas para entrenar
try:
    from src.pipeline_train import run_training_pipeline
    from src import config as cfg
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos desde 'src': {e}", file=sys.stderr)
    sys.exit(1)

def main_training(args):
    """
    Punto de entrada dedicado solo para el entrenamiento.
    Ahora usa 'args' en lugar de 'cfg' para los parámetros clave.
    """
    print("================================================")
    print("🔬 INICIANDO TRABAJO DE ENTRENAMIENTO (MODO WORKER) 🔬")
    print("================================================")
    
    # --- ¡NUEVO! Sobrescribir el config con los argumentos ---
    # Esto permite que el "Laboratorio" controle la configuración
    cfg.EXPERIMENT_NAME = args.name
    cfg.HIDDEN_CHANNELS = args.hidden_channels
    cfg.LR_RATE_M = args.lr
    cfg.EPISODES_TO_ADD = args.episodes
    
    # Seleccionar el modelo
    if args.model == 'unet':
        cfg.ACTIVE_MODEL_KEY = 'unet'
    else:
        cfg.ACTIVE_MODEL_KEY = 'mlp'
    # -------------------------------------------------

    try:
        # Llamar a la función de pipeline (que ahora leerá el cfg actualizado)
        run_training_pipeline()
        print("\n✅ Entrenamiento finalizado con éxito.")
    except Exception as e:
        print(f"\n❌ El entrenamiento falló con un error crítico: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento interrumpido por el usuario.")

# ==============================================================================
# PUNTO DE ENTRADA DEL SCRIPT DE ENTRENAMIENTO
# ==============================================================================
if __name__ == "__main__":
    # --- ¡NUEVO! Definir los argumentos de línea de comandos ---
    parser = argparse.ArgumentParser(description="Lanzador de Entrenamiento AETHERIA")
    parser.add_argument('--name', type=str, default=cfg.EXPERIMENT_NAME, help='Nombre de la carpeta del experimento')
    parser.add_argument('--model', type=str, default='unet', choices=['mlp', 'unet'], help='Modelo a entrenar (mlp o unet)')
    parser.add_argument('--hidden_channels', type=int, default=cfg.HIDDEN_CHANNELS, help='Canales ocultos (ancho) del modelo')
    parser.add_argument('--lr', type=float, default=cfg.LR_RATE_M, help='Tasa de aprendizaje')
    parser.add_argument('--episodes', type=int, default=cfg.EPISODES_TO_ADD, help='Número de episodios a entrenar')
    
    args = parser.parse_args()
    
    main_training(args)