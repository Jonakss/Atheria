# src/engines/base_engine.py
"""
EngineProtocol - Interfaz estandarizada para todos los motores de Atheria.

Todos los engines (Cartesian, Polar, Harmonic, Lattice, Native) deben
implementar esta interfaz para garantizar compatibilidad con:
- Pipeline de visualización
- Sistema de herramientas (tools)
- Frontend WebSocket
- Entrenamiento

FILOSOFÍA:
- Las estructuras EMERGEN de la evolución del campo, no se inyectan
- El estado es un tensor denso [B, H, W, C] (Channels-Last)
- La visualización espera datos normalizados [0, 1]
"""
from typing import Protocol, Optional, Dict, Any, Union, runtime_checkable
import torch
import numpy as np


@runtime_checkable
class EngineProtocol(Protocol):
    """
    Protocolo que define la interfaz mínima que todo engine debe implementar.
    
    Usar @runtime_checkable permite verificar en runtime con isinstance().
    """
    
    # Atributos requeridos
    device: Any
    grid_size: int
    d_state: int
    
    # ---------------------
    # EVOLUCIÓN
    # ---------------------
    
    def evolve_internal_state(self, step: Optional[int] = None) -> None:
        """
        Evoluciona el estado interno un paso de tiempo.
        
        Este es el método principal llamado por el loop de simulación.
        Modifica self.state in-place.
        
        Args:
            step: Número de paso actual (opcional, para logging/debugging)
        """
        ...
    
    def evolve_step(self, current_psi: torch.Tensor) -> torch.Tensor:
        """
        Evoluciona un estado dado y retorna el nuevo estado.
        
        A diferencia de evolve_internal_state, este método:
        - Recibe el estado como argumento
        - Retorna el nuevo estado (no modifica in-place)
        - Es útil para entrenamiento con gradientes
        
        Args:
            current_psi: Estado actual [B, H, W, C] complejo
            
        Returns:
            Nuevo estado [B, H, W, C] complejo
        """
        ...
    
    # ---------------------
    # ESTADO
    # ---------------------
    
    @property
    def state(self) -> Any:
        """
        Retorna el objeto de estado actual.
        
        El objeto debe tener un atributo `.psi` que sea un tensor complejo
        con shape [B, H, W, C].
        """
        ...
    
    @state.setter
    def state(self, new_state: Any) -> None:
        """
        Establece un nuevo estado.
        
        IMPORTANTE: Este setter puede regenerar el estado interno
        según la filosofía de EMERGENCIA (no inyectar partículas).
        """
        ...
    
    def get_dense_state(
        self, 
        roi: Optional[tuple] = None, 
        check_pause_callback: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Retorna el estado denso del sistema para visualización.
        
        Este método es crucial para:
        - Pipeline de visualización
        - Shaders del frontend
        - Análisis de estado
        
        Args:
            roi: Región de interés opcional (x, y, width, height)
            check_pause_callback: Callback para verificar si se debe pausar
                durante conversiones largas (usado por Native Engine)
        
        Returns:
            Tensor complejo [B, H, W, C] o [H, W, C]
        """
        ...
    
    def get_initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        Genera un estado inicial para entrenamiento.
        
        Args:
            batch_size: Tamaño del batch
            
        Returns:
            Tensor inicial [B, C, H, W] o [B, H, W, C] según el engine
        """
        ...
    
    # ---------------------
    # VISUALIZACIÓN
    # ---------------------
    
    def get_visualization_data(self, viz_type: str = "density") -> Dict[str, Any]:
        """
        Retorna datos de visualización normalizados para shaders.
        
        Los datos DEBEN estar normalizados a [0, 1] antes de retornar.
        
        Args:
            viz_type: Tipo de visualización
                - "density": |ψ|² (magnitud cuadrada)
                - "phase": arg(ψ) (fase)
                - "energy": Densidad de Hamiltoniano
                - "real": Re(ψ)
                - "imag": Im(ψ)
        
        Returns:
            dict con:
                - "data": np.ndarray normalizado [0, 1]
                - "type": str
                - "shape": list
                - "min": 0.0 (ya normalizado)
                - "max": 1.0 (ya normalizado)
                - "engine": str (nombre del engine)
        """
        ...
    
    # ---------------------
    # HERRAMIENTAS
    # ---------------------
    
    def apply_tool(self, action: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Aplica una herramienta cuántica al estado.
        
        NOTA: Estas herramientas son para interacción del usuario.
        El flujo normal de evolución NO usa este método.
        
        Args:
            action: Nombre de la acción ("collapse", "vortex", "wave", etc.)
            params: Parámetros de la acción (x, y, intensity, radius, etc.)
        
        Returns:
            Resultado opcional de la acción
        """
        ...
    
    # ---------------------
    # MODELO
    # ---------------------
    
    def compile_model(self) -> None:
        """
        Compila el modelo para optimización (torch.compile).
        
        Para engines sin modelo (Lattice), este es un no-op.
        """
        ...
    
    def get_model_for_params(self) -> Optional[Any]:
        """
        Retorna el modelo para acceder a sus parámetros.
        
        Útil para:
        - Contar parámetros
        - Guardar/cargar checkpoints
        - Optimizador
        
        Returns:
            El modelo nn.Module o None si no tiene modelo
        """
        ...


# ---------------------
# UTILIDADES
# ---------------------

def verify_engine_protocol(engine: Any, raise_on_error: bool = True) -> Dict[str, bool]:
    """
    Verifica que un engine implemente el EngineProtocol.
    
    Args:
        engine: Instancia del engine a verificar
        raise_on_error: Si True, lanza error si faltan métodos
    
    Returns:
        Dict con el estado de cada método requerido
    """
    required_methods = [
        'evolve_internal_state',
        'evolve_step', 
        'get_dense_state',
        'get_initial_state',
        'get_visualization_data',
        'apply_tool',
        'compile_model',
        'get_model_for_params',
    ]
    
    required_attrs = ['device', 'grid_size', 'd_state']
    
    results = {}
    missing = []
    
    for method in required_methods:
        has_method = hasattr(engine, method) and callable(getattr(engine, method))
        results[method] = has_method
        if not has_method:
            missing.append(method)
    
    for attr in required_attrs:
        has_attr = hasattr(engine, attr)
        results[attr] = has_attr
        if not has_attr:
            missing.append(attr)
    
    if missing and raise_on_error:
        engine_name = engine.__class__.__name__
        raise NotImplementedError(
            f"{engine_name} no implementa EngineProtocol. "
            f"Faltan: {', '.join(missing)}"
        )
    
    return results
