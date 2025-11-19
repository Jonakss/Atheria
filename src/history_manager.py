# src/history_manager.py
"""
Gestor de historia de simulación para análisis posterior.

Permite guardar y cargar estados de simulación para análisis offline.
"""
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import torch

HISTORY_DIR = Path("output/simulation_history")

class SimulationHistory:
    """Gestiona el historial de simulación."""
    
    def __init__(self, max_frames: int = 1000):
        """
        Args:
            max_frames: Número máximo de frames a mantener en memoria
        """
        self.max_frames = max_frames
        self.frames: List[Dict] = []
        self.history_dir = HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)
    
    def add_frame(self, frame_data: Dict):
        """
        Añade un frame al historial.
        
        Args:
            frame_data: Dict con datos del frame (step, map_data, hist_data, etc.)
        """
        # Optimizar: solo guardar datos esenciales y reducir tamaño
        map_data = frame_data.get('map_data')
        
        # Aplicar downsampling al map_data si es muy grande para reducir memoria
        # Guardar cada 2x2 píxeles como 1 píxel (reducción de 4x en memoria)
        if map_data is not None:
            try:
                if isinstance(map_data, np.ndarray):
                    # Downsample si el array es mayor a 128x128
                    if map_data.shape[0] > 128 or (len(map_data.shape) > 1 and map_data.shape[1] > 128):
                        # Downsample tomando cada 2x2 bloque y promediando
                        if len(map_data.shape) == 2:
                            # Array 2D: [height, width]
                            h, w = map_data.shape
                            downsampled = map_data[::2, ::2]  # Tomar cada 2 píxeles
                            map_data = downsampled.tolist() if not isinstance(map_data, list) else downsampled
                        elif isinstance(map_data, list) and len(map_data) > 0:
                            # Lista de listas: [[row1], [row2], ...]
                            if len(map_data) > 128 or (len(map_data[0]) if map_data else 0) > 128:
                                # Downsample manualmente
                                downsampled = []
                                for i in range(0, len(map_data), 2):
                                    row = map_data[i]
                                    if isinstance(row, list):
                                        downsampled_row = [row[j] for j in range(0, len(row), 2)]
                                        downsampled.append(downsampled_row)
                                    else:
                                        downsampled.append(row)
                                map_data = downsampled
            except Exception as e:
                logging.debug(f"Error aplicando downsampling al historial: {e}")
                # Si falla el downsampling, usar los datos originales
        
        optimized_frame = {
            'step': frame_data.get('step', 0),
            'timestamp': frame_data.get('timestamp', datetime.now().isoformat()),
            'map_data': map_data,  # Ya optimizado con downsampling si es necesario
            'hist_data': frame_data.get('hist_data', {}),
            # No guardar poincare_coords, phase_attractor, flow_data (se pueden recalcular)
        }
        
        self.frames.append(optimized_frame)
        
        # Limitar tamaño - eliminar los más antiguos si excede el límite
        if len(self.frames) > self.max_frames:
            # Eliminar los frames más antiguos (mantener solo los últimos max_frames)
            self.frames = self.frames[-self.max_frames:]
    
    def save_to_file(self, filename: Optional[str] = None) -> Path:
        """
        Guarda el historial a un archivo JSON.
        
        Args:
            filename: Nombre del archivo (opcional, se genera automáticamente si no se proporciona)
        
        Returns:
            Path del archivo guardado
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_history_{timestamp}.json"
        
        filepath = self.history_dir / filename
        
        # Convertir numpy arrays a listas para JSON
        serializable_frames = []
        for frame in self.frames:
            serializable_frame = {
                'step': frame['step'],
                'timestamp': frame['timestamp'],
                'map_data': frame['map_data'].tolist() if isinstance(frame['map_data'], np.ndarray) else frame['map_data'],
                'hist_data': frame['hist_data']
            }
            serializable_frames.append(serializable_frame)
        
        data = {
            'metadata': {
                'total_frames': len(serializable_frames),
                'created_at': datetime.now().isoformat(),
                'max_frames': self.max_frames
            },
            'frames': serializable_frames
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Historial guardado: {filepath} ({len(self.frames)} frames)")
        return filepath
    
    def load_from_file(self, filepath: Path) -> bool:
        """
        Carga un historial desde un archivo.
        
        Args:
            filepath: Path del archivo a cargar
        
        Returns:
            True si se cargó exitosamente, False en caso contrario
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.frames = data.get('frames', [])
            logging.info(f"Historial cargado: {filepath} ({len(self.frames)} frames)")
            return True
        except Exception as e:
            logging.error(f"Error cargando historial: {e}")
            return False
    
    def clear(self):
        """Limpia el historial."""
        self.frames = []
        logging.info("Historial limpiado")
    
    def get_frame(self, step: int) -> Optional[Dict]:
        """Obtiene un frame por su step."""
        for frame in self.frames:
            if frame['step'] == step:
                return frame
        return None
    
    def get_frames_range(self, start_step: int, end_step: int) -> List[Dict]:
        """Obtiene frames en un rango de steps."""
        return [f for f in self.frames if start_step <= f['step'] <= end_step]
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del historial."""
        if not self.frames:
            return {}
        
        steps = [f['step'] for f in self.frames]
        return {
            'total_frames': len(self.frames),
            'min_step': min(steps),
            'max_step': max(steps),
            'step_range': max(steps) - min(steps) if steps else 0
        }

