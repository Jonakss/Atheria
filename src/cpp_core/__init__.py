"""
Atheria Core: Módulo C++ de alto rendimiento.

Este módulo proporciona extensiones C++ compiladas para operaciones
de alto rendimiento en Atheria 4.

El módulo atheria_core se carga dinámicamente cuando se compila.
"""

"""
Atheria Core: Módulo C++ de alto rendimiento.

Este módulo proporciona extensiones C++ compiladas para operaciones
de alto rendimiento en Atheria 4.

El módulo atheria_core se carga dinámicamente cuando se compila.
"""

try:
    # Intentar importar el módulo compilado
    # El módulo se instala directamente en el path de Python
    import atheria_core as _core
    
    # Re-exportar componentes principales
    SparseMap = _core.SparseMap
    add = _core.add
    
    __all__ = ['SparseMap', 'add', '_core']
    
except ImportError:
    # Si el módulo no está compilado, proporcionar mensaje informativo
    import warnings
    warnings.warn(
        "atheria_core no está compilado. "
        "Compila el proyecto con 'pip install -e .' o 'python setup.py build_ext'",
        ImportWarning
    )
    
    __all__ = []

