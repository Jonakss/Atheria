# src/trainers/__init__.py
"""
Módulo de entrenadores para Atheria 4.
Exporta las clases de entrenadores disponibles.
"""

from .qc_trainer_v3 import QC_Trainer_v3
from .qc_trainer_v4 import QC_Trainer_v4

__all__ = ['QC_Trainer_v3', 'QC_Trainer_v4']

