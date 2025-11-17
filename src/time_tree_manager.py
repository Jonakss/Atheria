# src/time_tree_manager.py
"""
Sistema de "Árbol de Tiempo" para almacenar historia de simulación 2D de forma eficiente.

En lugar de guardar frames completos, guardamos:
- Keyframes completos cada N frames
- Deltas (diferencias) entre keyframes
- Estructura jerárquica simple basada en intervalos de tiempo

Esto permite navegación temporal eficiente. Se puede combinar con BinaryQuadtree
para compresión espacial adicional.
"""
import json
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from . import config as global_cfg


class TimeTreeManager:
    """
    Gestiona el almacenamiento eficiente de historia usando keyframes y deltas.
    """
    
    def __init__(self, experiment_name: str, keyframe_interval: int = 10, max_delta_size: float = 0.1):
        """
        Args:
            experiment_name: Nombre del experimento
            keyframe_interval: Cada cuántos frames guardar un keyframe completo
            max_delta_size: Tamaño máximo relativo de un delta (para compresión)
        """
        self.experiment_name = experiment_name
        self.keyframe_interval = keyframe_interval
        self.max_delta_size = max_delta_size
        
        # Directorio para almacenar el árbol de tiempo
        self.tree_dir = os.path.join(global_cfg.OUTPUT_DIR, "time_trees", experiment_name)
        os.makedirs(self.tree_dir, exist_ok=True)
        
        # Metadata del árbol
        self.metadata_path = os.path.join(self.tree_dir, "metadata.json")
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Carga los metadatos del árbol de tiempo."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Error cargando metadata del árbol de tiempo: {e}")
        
        return {
            "keyframes": [],  # Lista de índices de keyframes
            "deltas": [],     # Lista de deltas entre keyframes
            "total_frames": 0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """Guarda los metadatos del árbol de tiempo."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Error guardando metadata del árbol de tiempo: {e}")
    
    def _calculate_delta(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[Dict]:
        """
        Calcula el delta entre dos frames.
        
        Retorna None si el delta es demasiado grande (mejor guardar keyframe completo).
        """
        delta = frame2 - frame1
        
        # Calcular tamaño relativo del delta
        frame1_norm = np.linalg.norm(frame1)
        delta_norm = np.linalg.norm(delta)
        
        if frame1_norm > 0:
            relative_size = delta_norm / frame1_norm
            if relative_size > self.max_delta_size:
                # Delta demasiado grande, mejor usar keyframe
                return None
        
        # Guardar solo las posiciones donde hay cambios significativos
        threshold = np.max(np.abs(frame1)) * 0.01  # 1% del máximo
        changed_indices = np.where(np.abs(delta) > threshold)
        
        if len(changed_indices[0]) == 0:
            # No hay cambios significativos
            return {"type": "empty", "indices": []}
        
        # Guardar solo los cambios
        return {
            "type": "sparse",
            "indices": [int(i) for i in changed_indices[0]],  # Flatten indices
            "values": delta[changed_indices].tolist()
        }
    
    def _save_keyframe(self, frame_index: int, frame_data: np.ndarray):
        """Guarda un keyframe completo."""
        keyframe_path = os.path.join(self.tree_dir, f"keyframe_{frame_index:06d}.json")
        
        # Convertir a lista para JSON
        frame_list = frame_data.tolist()
        
        data = {
            "frame_index": frame_index,
            "timestamp": datetime.now().isoformat(),
            "data": frame_list
        }
        
        try:
            with open(keyframe_path, 'w') as f:
                json.dump(data, f)
            
            # Actualizar metadata
            if frame_index not in self.metadata["keyframes"]:
                self.metadata["keyframes"].append(frame_index)
                self.metadata["keyframes"].sort()
            
            logging.debug(f"Keyframe guardado: {keyframe_path}")
        except Exception as e:
            logging.error(f"Error guardando keyframe: {e}")
    
    def _save_delta(self, from_index: int, to_index: int, delta: Dict):
        """Guarda un delta entre dos frames."""
        delta_path = os.path.join(self.tree_dir, f"delta_{from_index:06d}_to_{to_index:06d}.json")
        
        data = {
            "from_index": from_index,
            "to_index": to_index,
            "delta": delta,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(delta_path, 'w') as f:
                json.dump(data, f)
            
            # Actualizar metadata
            delta_entry = {
                "from": from_index,
                "to": to_index,
                "path": delta_path
            }
            self.metadata["deltas"].append(delta_entry)
            
            logging.debug(f"Delta guardado: {delta_path}")
        except Exception as e:
            logging.error(f"Error guardando delta: {e}")
    
    def add_frame(self, frame_index: int, frame_data: np.ndarray):
        """
        Agrega un frame al árbol de tiempo.
        
        Decide si guardar como keyframe o como delta basado en el intervalo.
        """
        frame_array = np.array(frame_data)
        self.metadata["total_frames"] = max(self.metadata["total_frames"], frame_index + 1)
        
        # Si es múltiplo del intervalo, guardar como keyframe
        if frame_index % self.keyframe_interval == 0:
            self._save_keyframe(frame_index, frame_array)
            self._save_metadata()
            return
        
        # Buscar el keyframe más cercano anterior
        keyframes = self.metadata["keyframes"]
        if not keyframes:
            # No hay keyframes, guardar este como el primero
            self._save_keyframe(frame_index, frame_array)
            self._save_metadata()
            return
        
        # Encontrar el keyframe anterior más cercano
        prev_keyframe = None
        for kf in reversed(keyframes):
            if kf < frame_index:
                prev_keyframe = kf
                break
        
        if prev_keyframe is None:
            # Este frame es anterior a todos los keyframes, guardar como keyframe
            self._save_keyframe(frame_index, frame_array)
            self._save_metadata()
            return
        
        # Calcular delta desde el keyframe anterior
        prev_keyframe_data = self.get_frame(prev_keyframe)
        if prev_keyframe_data is not None:
            delta = self._calculate_delta(prev_keyframe_data, frame_array)
            
            if delta is None:
                # Delta demasiado grande, guardar como keyframe
                self._save_keyframe(frame_index, frame_array)
            else:
                # Guardar delta
                self._save_delta(prev_keyframe, frame_index, delta)
            
            self._save_metadata()
    
    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Reconstruye un frame desde el árbol de tiempo.
        
        Busca el keyframe más cercano y aplica deltas hasta llegar al frame deseado.
        """
        # Buscar keyframe más cercano anterior
        keyframes = self.metadata["keyframes"]
        if not keyframes:
            return None
        
        # Encontrar keyframe base
        base_keyframe = None
        for kf in reversed(keyframes):
            if kf <= frame_index:
                base_keyframe = kf
                break
        
        if base_keyframe is None:
            # No hay keyframe anterior, usar el primero
            base_keyframe = keyframes[0]
        
        # Cargar keyframe base
        keyframe_path = os.path.join(self.tree_dir, f"keyframe_{base_keyframe:06d}.json")
        if not os.path.exists(keyframe_path):
            logging.error(f"Keyframe no encontrado: {keyframe_path}")
            return None
        
        try:
            with open(keyframe_path, 'r') as f:
                keyframe_data = json.load(f)
            
            frame = np.array(keyframe_data["data"])
            
            # Si el frame deseado es el keyframe, retornar directamente
            if base_keyframe == frame_index:
                return frame
            
            # Aplicar deltas desde el keyframe hasta el frame deseado
            # Buscar todos los deltas entre base_keyframe y frame_index
            deltas_to_apply = [
                d for d in self.metadata["deltas"]
                if d["from"] >= base_keyframe and d["to"] <= frame_index
            ]
            
            # Ordenar por índice
            deltas_to_apply.sort(key=lambda x: x["to"])
            
            # Aplicar cada delta
            for delta_entry in deltas_to_apply:
                delta_path = delta_entry.get("path")
                if delta_path and os.path.exists(delta_path):
                    try:
                        with open(delta_path, 'r') as f:
                            delta_data = json.load(f)
                        
                        delta_info = delta_data["delta"]
                        
                        if delta_info["type"] == "empty":
                            continue
                        
                        if delta_info["type"] == "sparse":
                            # Aplicar cambios sparse
                            indices = delta_info["indices"]
                            values = delta_info["values"]
                            
                            # Reconstruir índices 2D desde índices planos
                            shape = frame.shape
                            for idx, val in zip(indices, values):
                                # Convertir índice plano a coordenadas 2D
                                row = idx // shape[1] if len(shape) > 1 else 0
                                col = idx % shape[1] if len(shape) > 1 else idx
                                
                                if row < shape[0] and (len(shape) == 1 or col < shape[1]):
                                    if len(shape) == 1:
                                        frame[idx] += val
                                    else:
                                        frame[row, col] += val
                    except Exception as e:
                        logging.warning(f"Error aplicando delta {delta_path}: {e}")
            
            return frame
            
        except Exception as e:
            logging.error(f"Error cargando keyframe: {e}")
            return None
    
    def get_all_frames(self) -> List[Tuple[int, np.ndarray]]:
        """Obtiene todos los frames disponibles (solo keyframes por ahora)."""
        frames = []
        for kf_index in self.metadata["keyframes"]:
            frame = self.get_frame(kf_index)
            if frame is not None:
                frames.append((kf_index, frame))
        return frames
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del árbol de tiempo."""
        keyframe_count = len(self.metadata["keyframes"])
        delta_count = len(self.metadata["deltas"])
        
        # Calcular tamaño total
        total_size = 0
        for file in os.listdir(self.tree_dir):
            file_path = os.path.join(self.tree_dir, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
        
        return {
            "total_frames": self.metadata["total_frames"],
            "keyframes": keyframe_count,
            "deltas": delta_count,
            "compression_ratio": self.metadata["total_frames"] / keyframe_count if keyframe_count > 0 else 1.0,
            "total_size_bytes": total_size,
            "created_at": self.metadata.get("created_at"),
            "last_updated": self.metadata.get("last_updated")
        }
    
    def clear(self):
        """Limpia todos los datos del árbol de tiempo."""
        try:
            import shutil
            shutil.rmtree(self.tree_dir)
            os.makedirs(self.tree_dir, exist_ok=True)
            self.metadata = self._load_metadata()
            logging.info(f"Árbol de tiempo limpiado para {self.experiment_name}")
        except Exception as e:
            logging.error(f"Error limpiando árbol de tiempo: {e}")

