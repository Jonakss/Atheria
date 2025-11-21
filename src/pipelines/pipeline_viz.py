# src/pipeline_viz.py
"""
Wrapper de compatibilidad para pipeline_viz.
El código real está en src/pipelines/viz/ para mejor organización modular.

Este archivo se mantiene por compatibilidad hacia atrás.
"""
from .viz import get_visualization_data

# Re-exportar para compatibilidad
__all__ = ['get_visualization_data']
