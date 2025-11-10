# src/config.py
import torch
import os

# --- Ruta Base del Proyecto ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------
# 0.1: SETUP Y CONSTANTES DE CONTROL
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directorios de Salida ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "training_checkpoints")
LARGE_SIM_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "simulation_checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LARGE_SIM_CHECKPOINT_DIR, exist_ok=True)

# --- Constantes del Servidor WebSocket ---
WEBSOCKET_HOST = '0.0.0.0'
WEBSOCKET_PORT = 8765

# --- Constantes del Servidor de Laboratorio ---
LAB_SERVER_HOST = '0.0.0.0'
LAB_SERVER_PORT = 8000

# ==============================================================================
# --- FASE 3: PARÁMETROS GLOBALES Y CONFIGURACIÓN ---
# ==============================================================================

# ------------------------------------------------------------------------------
# G. NOMBRE DEL EXPERIMENTO
# ------------------------------------------------------------------------------
EXPERIMENT_NAME = "Unitary_4D_v1" # <--- ¡Prueba con 4D primero!

# ------------------------------------------------------------------------------
# A. CONTROL DE EJECUCIÓN (Ajusta esto para entrenar)
# ------------------------------------------------------------------------------
RUN_TRAINING = True
RUN_POST_TRAINING_VIZ = False
RUN_LARGE_SIM = True
CONTINUE_TRAINING = False # False para empezar el nuevo experimento

# ------------------------------------------------------------------------------
# B. ARQUITECTURA DE LA LEY M (¡¡MODIFICADO!!)
# ------------------------------------------------------------------------------
GRID_SIZE_TRAINING = 64

# ¡¡NUEVO!! Especifica qué operador QCA usar.
# Opciones: "MLP", "UNET_UNITARIA"
ACTIVE_QCA_OPERATOR = "UNET_UNITARIA" 
MODEL_ARCHITECTURE = "UNET_UNITARIA"

# ¡¡NUEVO!! Dimensión del vector de estado REAL
# (Ya no usamos D_STATE=21)
# (Prueba con 4, 8, 16... 42)
D_STATE = 21 # Renombrado desde STATE_VECTOR_DIM para consistencia

# Ancho de la U-Net
HIDDEN_CHANNELS = 32

# ------------------------------------------------------------------------------
# C. PARÁMETROS DE ENTRENAMIENTO
# ------------------------------------------------------------------------------
EPISODES_TO_ADD = 2000
STEPS_PER_EPISODE = 50
LR_RATE_M = 1e-6 # (Bajo, bueno para U-Net)
PERSISTENCE_COUNT = 10
GRADIENT_CLIP = 0.85

# ------------------------------------------------------------------------------
# D. FUNCIÓN DE RECOMPENSA (¡¡MODIFICADA PARA FÍSICA UNITARIA!!)
# ------------------------------------------------------------------------------
# (El modelo unitario no puede "explotar", así que quitamos esas penalizaciones)

# Qué: (R_Quietud) Fomenta el "espacio vacío".
PESO_QUIETUD = 1.0

# Qué: (R_Complejidad) Fomenta la "materia".
PESO_COMPLEJIDAD_LOCALIZADA = 20.0

# ------------------------------------------------------------------------------
# E. PARÁMETROS DE REANUDACIÓN Y ESTANCAMIENTO
# ------------------------------------------------------------------------------
STAGNATION_WINDOW = 500
MIN_LOSS_IMPROVEMENT = 5e-6
REACTIVATION_COUNT = 2
REACTIVATION_STATE_MODE = 'random'
REACTIVATION_LR_MULTIPLIER = 0.5

# ------------------------------------------------------------------------------
# F. PARÁMETROS DE SIMULACIÓN Y VISUALIZACIÓN
# ------------------------------------------------------------------------------
GRID_SIZE = 512 # Tamaño de la grilla para servidores de visualización
SAVE_EVERY_EPISODES = 50
NUM_FRAMES_VIZ = 1500
FPS_VIZ_TRAINING = 24
GRID_SIZE_INFERENCE = 64
INITIAL_STATE_MODE_INFERENCE = 'complex_noise'
LOAD_STATE_CHECKPOINT_INFERENCE = True
STATE_CHECKPOINT_PATH_INFERENCE = ""
LARGE_SIM_CHECKPOINT_INTERVAL = 10000 
REAL_TIME_VIZ_INTERVAL = 5
REAL_TIME_VIZ_TYPE = 'change'
REAL_TIME_VIZ_DOWNSCALE = 2