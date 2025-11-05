# src/config.py
import torch
import os

# --- Ruta Base del Proyecto ---
# (Asume que config.py está en /src, así que subimos un nivel)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------
# 0.1: SETUP Y CONSTANTES DE CONTROL
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directorios de Salida (¡MODIFICADO!) ---
# Todas las salidas ahora van a la carpeta /output en la raíz
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "training_checkpoints")
LARGE_SIM_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "simulation_checkpoints")

# Crear directorios si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LARGE_SIM_CHECKPOINT_DIR, exist_ok=True)

# --- Constantes del Servidor WebSocket ---
WEBSOCKET_HOST = '0.0.0.0' # Escuchar en todas las interfaces
WEBSOCKET_PORT = 8765      # Puerto para la conexión

# ------------------------------------------------------------------------------
# 0.3: CONFIGURACIÓN OPCIONAL DEL MODELO DE ENTRADA
# ------------------------------------------------------------------------------
USE_INPUT_MODEL = False
INPUT_MODEL_PATH = "" 
USING_INPUT_MODEL_FLAG = False
if USE_INPUT_MODEL and INPUT_MODEL_PATH and os.path.exists(INPUT_MODEL_PATH):
    USING_INPUT_MODEL_FLAG = True
USE_INPUT_MODEL_FOR_TRAINING_RESUME = False


# ==============================================================================
# --- FASE 3: PARÁMETROS GLOBALES Y CONFIGURACIÓN ---
# ==============================================================================

# --- Control de Ejecución ---
RUN_TRAINING = False      # Poner en True para ejecutar la fase de entrenamiento.
RUN_POST_TRAINING_VIZ = False # Poner en True para la visualización post-entrenamiento.
RUN_LARGE_SIM = True     # Poner en True para ejecutar la simulación grande.

CONTINUE_TRAINING = False   # Poner en True para reanudar el entrenamiento.

# --- Parámetros de Simulación (Entrenamiento) ---
GRID_SIZE_TRAINING = 256     # Tamaño de la cuadrícula para entrenamiento
D_STATE = 21               # Dimensión del estado cuántico
HIDDEN_CHANNELS = 64        # Canales en la red de la Ley-M

# --- Parámetros de Entrenamiento Optimizados ---
EPISODES_TO_ADD = 500      # Número de episodios de entrenamiento a ejecutar
STEPS_PER_EPISODE = 50       # Pasos de simulación por episodio
LR_RATE_M = 5e-4           # Tasa de aprendizaje para el optimizador
PERSISTENCE_COUNT = 10       # (k en BPTT-k)

# --- Parámetros de Recompensa (Annealing Optimizado) ---
ALPHA_START = 3.0          # Peso inicial para R_Density_Target
ALPHA_END = 30.0           # Peso final para R_Density_Target
GAMMA_START = 3.0          # Peso inicial para R_Stability
GAMMA_END = 0.6            # Peso final para R_Stability
BETA_CAUSALITY = 3.0       # Peso fijo para R_Causality

# --- Pesos para Nuevas Recompensas ---
LAMBDA_ACTIVITY_VAR = 1.0  # Peso para R_Activity_Var
LAMBDA_VELOCIDAD = 0.5     # Peso para R_Velocidad

# --- Parámetros de "Búsqueda de Objetivo" y Penalización ---
TARGET_STD_DENSITY = 1.2   # Desviación estándar objetivo para la densidad
EXPLOSION_THRESHOLD = 0.7  # Densidad máx. por celda para activar penalización
EXPLOSION_PENALTY_MULTIPLIER = 20.0 # Multiplicador para penalización por explosión

# --- Parámetros de Estancamiento ---
STAGNATION_WINDOW = 500    # Episodios sin mejora antes de estancamiento
MIN_LOSS_IMPROVEMENT = 5e-5  # Mejora mínima de pérdida requerida

# --- Parámetros de Reactivación ---
REACTIVATION_COUNT = 2     # Número de intentos de reactivación
REACTIVATION_STATE_MODE = 'random' # 'random', 'seeded', 'complex_noise'
REACTIVATION_LR_MULTIPLIER = 0.5 # Factor para multiplicar LR en reactivación

# --- Recorte de Gradiente ---
GRADIENT_CLIP = 0.85       # Umbral para recorte de gradiente

# --- Frecuencia de Checkpoints (Entrenamiento) ---
SAVE_EVERY_EPISODES = 50   # Guardar checkpoint de entrenamiento cada N episodios.

# --- Parámetros de Visualización Post-Entrenamiento ---
NUM_FRAMES_VIZ = 1500      # Número de pasos para el video de visualización
FPS_VIZ_TRAINING = 24      # FPS para el video de visualización

# --- Parámetros de Simulación Grande (Inferencia) ---
GRID_SIZE_INFERENCE = 512    # Tamaño de la cuadrícula para simulación grande
# NUM_INFERENCE_STEPS ya no es necesario, el servidor corre indefinidamente

# --- Configuración de Inicialización (Inferencia) ---
INITIAL_STATE_MODE_INFERENCE = 'complex_noise' # 'random', 'seeded', 'complex_noise'
LOAD_STATE_CHECKPOINT_INFERENCE = True # Cargar checkpoint de estado de simulación grande
STATE_CHECKPOINT_PATH_INFERENCE = "" # Ruta específica al checkpoint de estado

# --- Frecuencia de Checkpoints (Simulación Grande) ---
LARGE_SIM_CHECKPOINT_INTERVAL = 10000 # Guardar checkpoint de estado cada N pasos.

# --- Parámetros de Guardado de Video (Simulación Grande) ---
# (Estos se usan en pipeline_viz, no en el servidor en tiempo real)
VIDEO_FPS = 35             # FPS para los videos generados
VIDEO_SAVE_INTERVAL_STEPS = 2 # Guardar un frame de video cada N pasos
VIDEO_DOWNSCALE_FACTOR = 2   # Factor para reducir resolución (1 = sin reducción)
VIDEO_QUALITY = 8            # Calidad de video (0-51, menor es mejor)

# --- Parámetros de Visualización en Tiempo Real (Simulación Grande) ---
REAL_TIME_VIZ_INTERVAL = 5   # Mostrar un frame cada N pasos
REAL_TIME_VIZ_TYPE = 'magnitude' # 'density', 'channels', 'magnitude', 'phase', 'change'
REAL_TIME_VIZ_DOWNSCALE = 2   # Factor para reducir resolución para visualización