# src/config.py
import os
import logging

# --- Ruta Base del Proyecto ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST_PATH = os.path.join(PROJECT_ROOT, "frontend", "dist")

# --- Setup y Constantes de Control ---
_DEVICE = None
def get_device():
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Dispositivo PyTorch inicializado: {_DEVICE}")
    return _DEVICE
DEVICE = get_device()

# --- Directorios y Servidor ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
EXPERIMENTS_DIR = os.path.join(OUTPUT_DIR, "experiments")
TRAINING_CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "training_checkpoints")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(TRAINING_CHECKPOINTS_DIR, exist_ok=True)
LAB_SERVER_HOST = '0.0.0.0'
LAB_SERVER_PORT = 8000

# ==============================================================================
# --- PARÁMETROS GLOBALES DE EXPERIMENTO ---
# ==============================================================================

# --- Gestión de Experimentos ---
EXPERIMENT_NAME = 'UNET_32ch_D5_LR2e-5'
LOAD_FROM_EXPERIMENT = 'UNET_32ch_D5_LR2e-5'
CONTINUE_TRAINING = True

# --- Arquitectura de la Ley M ---
# Opciones: 'UNET_UNITARIA', 'SNN_UNET', 'MLP', 'DEEP_QCA', 'UNET'
MODEL_ARCHITECTURE = 'UNET_UNITARIA'

# --- ¡¡NUEVO!! Parámetros Específicos del Modelo ---
# Este diccionario contiene todos los hiperparámetros para la *construcción* del modelo.
MODEL_PARAMS = {
    'd_state': 8,
    'hidden_channels': 32,
    # Parámetros para SNN_UNET (ignorados por otros modelos)
    'alpha': 0.9,
    'beta': 0.85,
}

# --- Física del Sistema ---
# Poner a 0.0 para una simulación unitaria "perfecta" (sin decaimiento).
GAMMA_DECAY = 0.01

# Hiperparámetros de Entrenamiento
TOTAL_EPISODES = 2000 # ¡¡CAMBIO!!
STEPS_PER_EPISODE = 50
LR_RATE_M = 1e-4
GRADIENT_CLIP = 1.0
GAMMA_DECAY = 0.01
SAVE_EVERY_EPISODES = 50
BATCH_SIZE_TRAINING = 4
QCA_STEPS_TRAINING = 16

# --- Función de Recompensa ---
PESO_QUIETUD = 10.0
PESO_COMPLEJIDAD_LOCALIZADA = 1.0

# --- Parámetros de Simulación ---
GRID_SIZE_TRAINING = 64
GRID_SIZE_INFERENCE = 256
SAVE_EVERY_EPISODES = 50
BATCH_SIZE_TRAINING = 4
QCA_STEPS_TRAINING = 16