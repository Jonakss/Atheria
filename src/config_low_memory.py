"""
Configuración optimizada para GPUs con memoria limitada (<4 GiB).

Este módulo provee parámetros reducidos para entrenar en hardware con VRAM limitada.
"""

# Arquitectura reducida
HIDDEN_CHANNELS_LOW_MEM = 64  # Reducido de 128
D_STATE_LOW_MEM = 10          # Reducido de 14

# Entrenamiento optimizado
QCA_STEPS_LOW_MEM = 200       # Reducido de 500
GRID_SIZE_LOW_MEM = 64        # Mantener 64

# Optimizaciones de memoria
USE_GRADIENT_CHECKPOINTING = True
USE_MIXED_PRECISION = True    # Automatic Mixed Precision (AMP)
CUDA_MEMORY_FRACTION = 0.95   # Limitar memoria usable al 95%

# Caché más agresivo
EMPTY_CACHE_INTERVAL = 5      # Vaciar caché CUDA cada 5 episodios (antes: 10)

# Configuración de GPU
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
