# src/config.py
import os
import logging
import warnings

# --- Silenciar warnings ANTES de importar torch ---
# Los warnings de CUDA se emiten durante la importación de torch,
# por lo que debemos configurar los filtros primero
warnings.filterwarnings('ignore', message='.*cudagraph.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CUDA graph.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CUDA initialization.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*cudaGetDeviceCount.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*invalid device ordinal.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Unexpected error from.*', category=UserWarning)

# --- Ruta Base del Proyecto ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST_PATH = os.path.join(PROJECT_ROOT, "frontend", "dist")

# --- Importar torch con warnings silenciados ---
# Estos warnings son informativos y no afectan la funcionalidad
# PyTorch intenta optimizar con CUDA graphs pero algunas operaciones no son capturables
# El error 101 (invalid device ordinal) es común cuando CUDA no está disponible correctamente
import torch

# Configurar logging de PyTorch para reducir mensajes de CUDA graph
# Los mensajes de "cudagraph partition" son informativos y esperados cuando
# torch.compile encuentra operaciones no capturables (como torch.cat dinámico)
# Los warnings ya fueron silenciados antes de importar torch

# --- Setup y Constantes de Control ---
_DEVICE = None
def get_device():
    global _DEVICE
    if _DEVICE is None:
        # Intentar detectar CUDA con manejo de errores robusto
        cuda_available = False
        try:
            # Silenciar warnings durante la detección de CUDA
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                # Intentar acceder a CUDA de forma segura
                if hasattr(torch.cuda, 'is_available'):
                    cuda_available = torch.cuda.is_available()
                    # Verificar que realmente podemos usar CUDA (no solo que está disponible)
                    if cuda_available:
                        try:
                            # Intentar obtener el device count para verificar que funciona
                            device_count = torch.cuda.device_count()
                            if device_count == 0:
                                cuda_available = False
                        except (RuntimeError, AttributeError):
                            # Si falla al obtener device count, CUDA no es usable
                            cuda_available = False
        except Exception as e:
            # Si hay cualquier error, usar CPU como fallback
            logging.debug(f"Error detectando CUDA: {e}. Usando CPU como fallback.")
            cuda_available = False
        
        _DEVICE = torch.device("cuda" if cuda_available else "cpu")
        if cuda_available:
            logging.info(f"Dispositivo PyTorch inicializado: {_DEVICE}")
        else:
            logging.info(f"Dispositivo PyTorch inicializado: {_DEVICE} (CUDA no disponible o falló la inicialización)")
    return _DEVICE
DEVICE = get_device()

# --- Configuración del Motor Nativo C++ ---
# Parámetro para forzar el device del motor nativo:
# - "auto": Detectar automáticamente el mejor disponible (GPU si está disponible, sino CPU)
# - "cpu": Forzar CPU
# - "cuda": Forzar CUDA/GPU (fallará si no está disponible)
NATIVE_ENGINE_DEVICE = os.environ.get('ATHERIA_NATIVE_DEVICE', 'auto').lower()

def get_native_engine_device() -> str:
    """
    Obtiene el device para el motor nativo C++.
    
    Si NATIVE_ENGINE_DEVICE es "auto", detecta automáticamente el mejor disponible.
    Si es "cpu" o "cuda", retorna ese valor directamente.
    
    Returns:
        str: 'cpu' o 'cuda'
    """
    if NATIVE_ENGINE_DEVICE == 'auto':
        # Detectar automáticamente el mejor disponible
        # Intentar CUDA primero, luego CPU como fallback
        cuda_available = False
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if hasattr(torch.cuda, 'is_available'):
                    cuda_available = torch.cuda.is_available()
                    if cuda_available:
                        try:
                            device_count = torch.cuda.device_count()
                            if device_count == 0:
                                cuda_available = False
                        except (RuntimeError, AttributeError):
                            cuda_available = False
        except Exception:
            cuda_available = False
        
        selected_device = "cuda" if cuda_available else "cpu"
        if cuda_available:
            logging.info(f"Motor nativo: Auto-detectado CUDA (mejor disponible)")
        else:
            logging.info(f"Motor nativo: Auto-detectado CPU (CUDA no disponible)")
        return selected_device
    elif NATIVE_ENGINE_DEVICE in ('cpu', 'cuda'):
        # Forzar device específico
        if NATIVE_ENGINE_DEVICE == 'cuda':
            # Verificar que CUDA está disponible si se fuerza
            try:
                if not torch.cuda.is_available():
                    logging.warning(f"⚠️ Se forzó CUDA pero no está disponible. Usando CPU como fallback.")
                    return 'cpu'
                device_count = torch.cuda.device_count()
                if device_count == 0:
                    logging.warning(f"⚠️ Se forzó CUDA pero no hay dispositivos disponibles. Usando CPU como fallback.")
                    return 'cpu'
            except Exception as e:
                logging.warning(f"⚠️ Error verificando CUDA: {e}. Usando CPU como fallback.")
                return 'cpu'
        logging.info(f"Motor nativo: Device forzado a {NATIVE_ENGINE_DEVICE.upper()}")
        return NATIVE_ENGINE_DEVICE
    else:
        # Valor inválido, usar auto
        logging.warning(f"⚠️ Valor inválido para NATIVE_ENGINE_DEVICE: '{NATIVE_ENGINE_DEVICE}'. Usando 'auto'.")
        return get_native_engine_device.__wrapped__() if hasattr(get_native_engine_device, '__wrapped__') else 'cpu'

# Variable global para el device del motor nativo (se inicializa en primera llamada)
_NATIVE_ENGINE_DEVICE = None

def get_native_device():
    """Función helper para obtener el device del motor nativo (con caché)."""
    global _NATIVE_ENGINE_DEVICE
    if _NATIVE_ENGINE_DEVICE is None:
        _NATIVE_ENGINE_DEVICE = get_native_engine_device()
    return _NATIVE_ENGINE_DEVICE

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