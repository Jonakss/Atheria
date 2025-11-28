# src/cache/__init__.py
"""
Módulo de caché distribuido para Atheria.
Proporciona caché de alta velocidad usando Dragonfly.
"""
from .dragonfly_client import DragonflyCache, cache

__all__ = ['DragonflyCache', 'cache']
