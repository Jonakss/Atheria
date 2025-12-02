"""Helpers para construir mensajes de estado con compile_status."""
import logging
from typing import Dict, Any, Optional

from ...server.server_state import g_state


def get_compile_status() -> Optional[Dict[str, Any]]:
    """
    Obtiene el compile_status actual desde g_state o lo reconstruye desde el motor.
    
    Returns:
        Dict con compile_status o None si no hay motor cargado
    """
    # Primero intentar obtener desde g_state si está guardado
    if 'compile_status' in g_state and g_state['compile_status']:
        return g_state['compile_status']
    
    # Si no está en g_state, reconstruir desde el motor actual
    motor = g_state.get('motor')
    if not motor:
        return None
    
    is_native = g_state.get('motor_is_native', False)
    
    try:
        if is_native:
            # Motor nativo
            from ...engines.native_engine_wrapper import NativeEngineWrapper
            
            device_str = 'cuda' if g_state.get('device', 'cpu') == 'cuda' else 'cpu'
            native_version = getattr(motor, 'native_version', None) or "unknown"
            
            wrapper_version = None
            try:
                wrapper_version = getattr(NativeEngineWrapper, 'VERSION', None) or "unknown"
            except:
                wrapper_version = "unknown"
            
            compile_status = {
                "is_compiled": True,
                "is_native": True,
                "model_name": "Native Engine (C++)",
                "compiles_enabled": True,
                "device_str": device_str,
                "native_version": native_version,
                "wrapper_version": wrapper_version
            }
        else:
            # Motor Python
            from ...engines.qca_engine import CartesianEngine
            
            device_str = str(motor.device) if hasattr(motor, 'device') else 'cpu'
            if 'cuda' in device_str.lower():
                device_str = 'cuda'
            else:
                device_str = 'cpu'
            
            python_version = getattr(motor, 'VERSION', None) or (motor.get_version() if hasattr(motor, 'get_version') else 'unknown')
            
            model = g_state.get('model')
            model_name = model.__class__.__name__ if model and hasattr(model, '__class__') else "Unknown"
            is_compiled = getattr(motor, 'is_compiled', False) if hasattr(motor, 'is_compiled') else False
            
            compile_status = {
                "is_compiled": is_compiled,
                "is_native": False,
                "model_name": model_name,
                "compiles_enabled": getattr(model, '_compiles', True) if model and hasattr(model, '_compiles') else True,
                "device_str": device_str,
                "python_version": python_version
            }
        
        # Guardar en g_state para futuras referencias
        g_state['compile_status'] = compile_status
        return compile_status
        
    except Exception as e:
        logging.debug(f"Error obteniendo compile_status: {e}")
        return None


def build_inference_status_payload(status: str, **kwargs) -> Dict[str, Any]:
    """
    Construye un payload de inference_status_update con compile_status incluido.
    
    Args:
        status: Estado de la inferencia ('running' o 'paused')
        **kwargs: Campos adicionales para el payload (model_loaded, experiment_name, etc.)
    
    Returns:
        Dict con el payload completo incluyendo compile_status si está disponible
    """
    payload = {"status": status}
    payload.update(kwargs)
    
    # Intentar obtener compile_status
    compile_status = get_compile_status()
    if compile_status:
        payload["compile_status"] = compile_status
    
    return payload

