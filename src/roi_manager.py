# src/roi_manager.py
"""
Módulo para gestionar Region of Interest (ROI) - visualización de una región específica del grid.
Permite visualizar solo una parte del grid sin enviar todos los datos, optimizando la transferencia.
"""
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging

class ROIManager:
    """
    Gestiona la región de interés (ROI) para visualización.
    """
    
    def __init__(self, grid_size: int = 256):
        self.grid_size = grid_size
        self.roi_enabled = False
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = grid_size
        self.roi_height = grid_size
    
    def set_roi(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Establece la región de interés.
        
        Args:
            x: Coordenada X inicial (0-indexed)
            y: Coordenada Y inicial (0-indexed)
            width: Ancho de la ROI
            height: Alto de la ROI
        
        Returns:
            True si la ROI es válida, False en caso contrario
        """
        # Validar que la ROI esté dentro del grid
        if x < 0 or y < 0:
            return False
        if x + width > self.grid_size or y + height > self.grid_size:
            return False
        if width <= 0 or height <= 0:
            return False
        
        self.roi_x = x
        self.roi_y = y
        self.roi_width = width
        self.roi_height = height
        self.roi_enabled = True
        
        logging.info(f"ROI configurada: ({x}, {y}) tamaño {width}x{height}")
        return True
    
    def clear_roi(self):
        """Desactiva la ROI y muestra el grid completo."""
        self.roi_enabled = False
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = self.grid_size
        self.roi_height = self.grid_size
        logging.info("ROI desactivada, mostrando grid completo")
    
    def get_roi_slice(self) -> Tuple[slice, slice]:
        """
        Retorna los slices de NumPy para extraer la ROI.
        
        Returns:
            Tuple (slice_y, slice_x) para usar con arrays NumPy
        """
        if not self.roi_enabled:
            return slice(None), slice(None)
        
        return slice(self.roi_y, self.roi_y + self.roi_height), slice(self.roi_x, self.roi_x + self.roi_width)
    
    def extract_roi(self, data: np.ndarray) -> np.ndarray:
        """
        Extrae la región de interés de un array NumPy.
        
        Args:
            data: Array 2D o 3D (H, W) o (H, W, channels)
        
        Returns:
            Array recortado a la ROI
        """
        if not self.roi_enabled:
            return data
        
        slice_y, slice_x = self.get_roi_slice()
        
        if len(data.shape) == 2:
            # Array 2D (H, W)
            return data[slice_y, slice_x]
        elif len(data.shape) == 3:
            # Array 3D (H, W, channels)
            return data[slice_y, slice_x, :]
        else:
            logging.warning(f"Forma de array no soportada para ROI: {data.shape}")
            return data
    
    def get_roi_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre la ROI actual.
        
        Returns:
            Dict con información de la ROI
        """
        if not self.roi_enabled:
            return {
                "enabled": False,
                "x": 0,
                "y": 0,
                "width": self.grid_size,
                "height": self.grid_size,
                "area": self.grid_size * self.grid_size,
                "reduction_ratio": 1.0
            }
        
        total_area = self.grid_size * self.grid_size
        roi_area = self.roi_width * self.roi_height
        reduction_ratio = total_area / roi_area if roi_area > 0 else 1.0
        
        return {
            "enabled": True,
            "x": self.roi_x,
            "y": self.roi_y,
            "width": self.roi_width,
            "height": self.roi_height,
            "area": roi_area,
            "reduction_ratio": reduction_ratio,
            "total_area": total_area
        }

def apply_roi_to_payload(payload: Dict[str, Any], roi_manager: ROIManager) -> Dict[str, Any]:
    """
    Aplica la ROI a un payload de frame antes de enviarlo.
    
    Args:
        payload: Payload original con map_data, etc.
        roi_manager: Instancia de ROIManager
    
    Returns:
        Payload optimizado con solo la región de interés
    """
    if not roi_manager.roi_enabled:
        return payload
    
    optimized = payload.copy()
    slice_y, slice_x = roi_manager.get_roi_slice()
    
    # Aplicar ROI a map_data
    if 'map_data' in payload and payload['map_data']:
        try:
            if isinstance(payload['map_data'], list):
                map_data = np.array(payload['map_data'])
                roi_data = roi_manager.extract_roi(map_data)
                optimized['map_data'] = roi_data.tolist()
        except Exception as e:
            logging.warning(f"Error aplicando ROI a map_data: {e}")
    
    # Aplicar ROI a complex_3d_data
    if 'complex_3d_data' in payload and payload['complex_3d_data']:
        try:
            complex_data = payload['complex_3d_data']
            if 'real' in complex_data and isinstance(complex_data['real'], list):
                real_array = np.array(complex_data['real'])
                roi_real = roi_manager.extract_roi(real_array)
                optimized['complex_3d_data']['real'] = roi_real.tolist()
            
            if 'imag' in complex_data and isinstance(complex_data['imag'], list):
                imag_array = np.array(complex_data['imag'])
                roi_imag = roi_manager.extract_roi(imag_array)
                optimized['complex_3d_data']['imag'] = roi_imag.tolist()
        except Exception as e:
            logging.warning(f"Error aplicando ROI a complex_3d_data: {e}")
    
    # Aplicar ROI a flow_data
    if 'flow_data' in payload and payload['flow_data']:
        try:
            flow_data = payload['flow_data']
            for key in ['dx', 'dy', 'magnitude']:
                if key in flow_data and isinstance(flow_data[key], list):
                    flow_array = np.array(flow_data[key])
                    roi_flow = roi_manager.extract_roi(flow_array)
                    optimized['flow_data'][key] = roi_flow.tolist()
        except Exception as e:
            logging.warning(f"Error aplicando ROI a flow_data: {e}")
    
    # Aplicar ROI a phase_hsv_data
    if 'phase_hsv_data' in payload and payload['phase_hsv_data']:
        try:
            hsv_data = payload['phase_hsv_data']
            for key in ['hue', 'saturation', 'value']:
                if key in hsv_data and isinstance(hsv_data[key], list):
                    hsv_array = np.array(hsv_data[key])
                    roi_hsv = roi_manager.extract_roi(hsv_array)
                    optimized['phase_hsv_data'][key] = roi_hsv.tolist()
        except Exception as e:
            logging.warning(f"Error aplicando ROI a phase_hsv_data: {e}")
    
    # Añadir información de ROI al payload
    optimized['roi_info'] = roi_manager.get_roi_info()
    
    return optimized

