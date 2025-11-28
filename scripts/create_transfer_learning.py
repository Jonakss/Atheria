#!/usr/bin/env python3
"""
Script para crear un experimento de transfer learning con valores seguros.
Uso: python scripts/create_transfer_learning.py
"""
import sys
import os

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_experiment_config
from types import SimpleNamespace
from datetime import datetime

# Configuración del experimento con Transfer Learning
exp_config = {
    "EXPERIMENT_NAME": "UNET_UNITARY_RMSNORM-d14-h128-g32-lr5e-5-TL",
    "MODEL_ARCHITECTURE": "UNET_UNITARY_RMSNORM",
    
    # Transfer Learning desde UNET_UNITARY
    "LOAD_FROM_EXPERIMENT": "UNET_UNITARY-d14-h64-g32-lr8e-5",
    
    # Parámetros del modelo
    "MODEL_PARAMS": {
        "d_state": 14,           # Mismo que el modelo base
        "hidden_channels": 128,  # Aumentado de 64 para más capacidad
        "alpha": 0.9,
        "beta": 0.85
    },
    
    # Configuración de entrenamiento
    "GRID_SIZE_TRAINING": 32,      # Mismo que el modelo base
    "QCA_STEPS_TRAINING": 1000,    # Razonable para entrenamiento
    "TOTAL_EPISODES": 200,         # Suficiente para transfer learning
    "LR_RATE_M": 0.00005,          # 5e-5, más bajo que el base para fine-tuning
    "GAMMA_DECAY": 0.01,           # Término Lindbladian estándar
    
    # Modo de inicialización
    "INITIAL_STATE_MODE_INFERENCE": "complex_noise",
    
    # Metadata
    "created_at": datetime.now().isoformat(),
    "TRAINER_VERSION": "v4",
    "description": "Transfer learning desde UNET_UNITARY con RMSNORM para mayor velocidad"
}

if __name__ == "__main__":
    print("🚀 Creando experimento de Transfer Learning...")
    print(f"📦 Modelo base: {exp_config['LOAD_FROM_EXPERIMENT']}")
    print(f"🏗️  Nuevo modelo: {exp_config['EXPERIMENT_NAME']}")
    print(f"📊 Arquitectura: {exp_config['MODEL_ARCHITECTURE']}")
    print(f"📐 Grid: {exp_config['GRID_SIZE_TRAINING']}x{exp_config['GRID_SIZE_TRAINING']}")
    print(f"🔬 Hidden Channels: {exp_config['MODEL_PARAMS']['hidden_channels']}")
    print(f"📈 Learning Rate: {exp_config['LR_RATE_M']}")
    print(f"🎯 Episodios: {exp_config['TOTAL_EPISODES']}")
    
    try:
        save_experiment_config(exp_config['EXPERIMENT_NAME'], exp_config)
        print(f"\n✅ Configuración guardada exitosamente!")
        print(f"\n📝 Para iniciar el entrenamiento, ejecuta:")
        print(f"   python -m src.trainer --experiment_name {exp_config['EXPERIMENT_NAME']}")
    except Exception as e:
        print(f"\n❌ Error al guardar configuración: {e}")
        sys.exit(1)
