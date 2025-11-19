# src/config.py
import os
import logging
import warnings

# --- Ruta Base del Proyecto ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST_PATH = os.path.join(PROJECT_ROOT, "frontend", "dist")

# --- Silenciar warnings de CUDA graph de PyTorch ---
# Estos warnings son informativos y no afectan la funcionalidad
# PyTorch intenta optimizar con CUDA graphs pero algunas operaciones no son capturables
import torch

# Configurar logging de PyTorch para reducir mensajes de CUDA graph
# Los mensajes de "cudagraph partition" son informativos y esperados cuando
# torch.compile encuentra operaciones no capturables (como torch.cat dinámico)
try:
    # Silenciar mensajes de partición de CUDA graph vía variables de entorno
    import os
    # PYTORCH_CUDA_ALLOC_CONF no silencia estos mensajes, pero podemos usar logging
    # Los mensajes vienen directamente del backend de C++ y no se pueden silenciar fácilmente
    # Sin embargo, son informativos y no afectan el rendimiento o funcionalidad
    pass
except Exception:
    pass

# Configurar filtros de warnings para Python warnings relacionados
warnings.filterwarnings('ignore', message='.*cudagraph.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CUDA graph.*', category=UserWarning)

# --- Setup y Constantes de Control ---
_DEVICE = None
def get_device():
    global _DEVICE
    if _DEVICE is None:
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

# --- Física del Sistema (Ecuación Maestra de Lindblad) ---
# GAMMA_DECAY implementa el término Lindbladian para sistemas cuánticos abiertos.
# 
# Ecuación completa: dρ/dt = -i[H, ρ] + Σ_i γ_i (L_i ρ L_i† - (1/2){L_i† L_i, ρ})
# 
# Nuestra implementación simplificada:
#   - Parte 1 (Unitaria): dΨ/dt = A(Ψ) * Ψ (implementada por la Ley M / U-Net)
#   - Parte 2 (Lindbladian): dΨ/dt = -γ * Ψ (decaimiento/disipación)
# 
# GAMMA_DECAY controla la "presión evolutiva" hacia el metabolismo:
#   - 0.0 = Sistema cerrado perfecto (sin decaimiento, sin "hambre")
#   - > 0.0 = Sistema abierto (con decaimiento, la Ley M debe "ganar" contra el decaimiento)
# 
# Valores típicos:
#   - 0.0 - 0.001: Decaimiento muy lento (presión suave)
#   - 0.01 - 0.1: Decaimiento moderado (presión estándar para A-Life)
#   - > 0.1: Decaimiento rápido (presión fuerte, más difícil mantener estructuras)
GAMMA_DECAY = 0.01

# --- Motor Nativo (C++) ---
# Si es True, intenta usar el motor nativo de alto rendimiento (250-400x más rápido)
# Si es False o el motor nativo no está disponible, usa el motor Python tradicional
USE_NATIVE_ENGINE = True

# Hiperparámetros de Entrenamiento
TOTAL_EPISODES = 2000 # ¡¡CAMBIO!!
STEPS_PER_EPISODE = 50
LR_RATE_M = 1e-4
GRADIENT_CLIP = 1.0
# Nota: GAMMA_DECAY está definido arriba en "Física del Sistema"
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

# --- Modo de Inicialización del Estado para Inferencia ---
# Opciones: 'complex_noise', 'random', 'zeros'
# - 'complex_noise': Ruido complejo normalizado (default, más estable)
# - 'random': Estado aleatorio normalizado (más variado)
# - 'zeros': Estado cero (requiere activación externa)
INITIAL_STATE_MODE_INFERENCE = 'complex_noise'