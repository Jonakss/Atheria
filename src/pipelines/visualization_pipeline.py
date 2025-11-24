import asyncio
import logging
import torch
from typing import Dict, Any, Optional
from .viz.core import get_visualization_data

class VisualizationPipeline:
    """
    Pipeline de visualización que gestiona la generación de frames.
    Envuelve las funciones de visualización core y maneja la ejecución asíncrona.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("VisualizationPipeline")
        
    async def generate_frame(self, psi: torch.Tensor, step: int, viz_type: str = 'density', 
                           motor: Any = None) -> Optional[Dict[str, Any]]:
        """
        Genera un frame de visualización de manera asíncrona (en thread pool).
        
        Args:
            psi: Estado cuántico (tensor)
            step: Paso actual de simulación
            viz_type: Tipo de visualización
            motor: Referencia al motor (opcional)
            
        Returns:
            Dict con datos de visualización o None si falla
        """
        try:
            # Ejecutar en thread pool para no bloquear el event loop
            loop = asyncio.get_running_loop()
            
            # Wrapper para llamar a get_visualization_data con los argumentos correctos
            def _generate():
                return get_visualization_data(
                    psi=psi,
                    viz_type=viz_type,
                    motor=motor
                )
            
            viz_data = await loop.run_in_executor(None, _generate)
            
            # Agregar step al resultado si se generó correctamente
            if viz_data:
                viz_data['step'] = step
                
            return viz_data
            
        except Exception as e:
            self.logger.error(f"Error generando frame de visualización: {e}")
            return None
