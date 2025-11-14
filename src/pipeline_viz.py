# src/pipeline_viz.py
import logging
from . import visualization_tools as viz

# --- ¡¡NUEVO!! Mapa de Despacho de Visualizaciones ---
VIZ_MAP = {
    'density_map': viz.get_density_map,
    'channels_map': viz.get_channels_map,
    'phase_map': viz.get_phase_map,
    'change_map': viz.get_change_map,
}

def run_visualization_pipeline(motor, viz_type: str):
    """
    Ejecuta el pipeline de visualización para un motor y un tipo de mapa dados.
    """
    try:
        # 1. Ejecutar un paso de evolución en el motor (sin entrenamiento)
        motor.evolve_step(is_training=False)
        
        # 2. Obtener el estado cuántico actual
        psi = motor.state.psi
        
        # 3. Usar el mapa para obtener la función de visualización correcta
        viz_function = VIZ_MAP.get(viz_type)
        
        if not viz_function:
            logging.warning(f"Tipo de visualización desconocido: '{viz_type}'. Usando 'density_map' por defecto.")
            viz_function = viz.get_density_map

        # 4. Generar el mapa y convertirlo a una lista para JSON
        frame_map = viz_function(psi)
        
        return {
            "step": motor.state.psi.shape[0], # Placeholder para el número de paso
            "viz_type": viz_type,
            "map_data": frame_map.tolist()
        }

    except Exception as e:
        logging.error(f"Error en el pipeline de visualización: {e}", exc_info=True)
        return None
