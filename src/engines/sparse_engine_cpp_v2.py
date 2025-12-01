# src/engines/sparse_engine_cpp_v2.py
"""
Motor de simulación disperso usando el núcleo C++ atheria_core con tensores nativos.

Esta versión usa SparseMap de C++ con almacenamiento directo de torch::Tensor,
eliminando completamente la necesidad de diccionarios auxiliares en Python.

VERSIÓN 2: Usa tensores nativos de LibTorch almacenados directamente en C++.
"""
import torch
import numpy as np
import math

try:
    import atheria_core
    CPP_AVAILABLE = True
    TORCH_SUPPORT = atheria_core.has_torch_support()
except ImportError:
    CPP_AVAILABLE = False
    TORCH_SUPPORT = False
    import warnings
    warnings.warn("atheria_core no está disponible. Usando implementación Python pura.")


class QuantumVacuum:
    """
    Generador procedural del 'Estado Base' (Vacío Cuántico).
    En QFT, el vacío no es cero, es un estado de mínima energía con fluctuaciones.
    """
    def __init__(self, d_state, device):
        self.d_state = d_state
        self.device = device
        
    def get_fluctuation(self, coords, t):
        """
        Genera una fluctuación determinista (pseudo-aleatoria) para una coordenada.
        Esto asegura que el vacío sea consistente (si vuelves al mismo sitio, el 'ruido' es el mismo).
        """
        # Usamos hashing de coordenadas para generar una semilla determinista
        # Esto simula un campo de fondo estático o fluctuante pero coherente
        x, y = coords[0], coords[1]
        z = coords[2] if len(coords) > 2 else 0
        
        seed = hash((x, y, z, t)) % (2**32)
        torch.manual_seed(seed)
        
        # Generar ruido de muy baja amplitud (fluctuaciones de punto cero)
        noise = torch.randn(self.d_state, device=self.device) * 0.001
        return noise


class SparseQuantumEngineCppV2:
    """
    Motor de simulación para un universo potencialmente infinito usando C++ con tensores nativos.
    
    Esta versión usa atheria_core.SparseMap con almacenamiento directo de torch::Tensor
    en C++, eliminando completamente los diccionarios auxiliares de Python.
    
    VENTAJAS:
    - Almacenamiento nativo de tensores en C++ (torch::Tensor)
    - Sin diccionarios espejo en Python
    - Mejor gestión de memoria para grandes cantidades de datos
    - Preparado para operaciones vectorizadas futuras en C++
    
    NOTA: Para operaciones pequeñas puede ser más lento que Python dict debido a
    overhead de bindings, pero tiene mejor arquitectura para optimizaciones futuras.
    """
    def __init__(self, model, d_state, device='cpu', vacuum_mode='active'):
        self.model = model
        self.d_state = d_state
        self.device = device
        self.vacuum = QuantumVacuum(d_state, device)
        
        # El Universo: SparseMap C++ con tensores nativos
        if CPP_AVAILABLE and TORCH_SUPPORT:
            self.matter_map = atheria_core.SparseMap()
            self._use_cpp_native = True
        elif CPP_AVAILABLE:
            # Fallback a SparseMap sin tensores (usar dict auxiliar)
            self.matter_map = atheria_core.SparseMap()
            self._state_tensors = {}  # Diccionario auxiliar
            self._use_cpp_native = False
            import warnings
            warnings.warn("LibTorch no disponible. Usando SparseMap con diccionario auxiliar.")
        else:
            # Fallback completo a Python
            self.matter = {}
            self._use_cpp_native = False
            import warnings
            warnings.warn("atheria_core no disponible. Usando implementación Python pura.")
        
        # Conjunto de celdas que necesitan actualización en el siguiente paso
        # (Incluye materia y su vecindario inmediato)
        self.active_region = set()
        
        self.step_count = 0
    
    def _coord_to_key(self, coords):
        """Convierte coordenadas (x, y, z) a Coord3D"""
        x, y = coords[0], coords[1]
        z = coords[2] if len(coords) > 2 else 0
        return atheria_core.Coord3D(x, y, z)

    def add_particle(self, coords, state_vector):
        """Inyecta una excitación (partícula) en el campo."""
        if self._use_cpp_native:
            # Almacenamiento nativo en C++ (versión optimizada)
            coord = self._coord_to_key(coords)
            self.matter_map.insert_tensor(coord, state_vector)
        elif hasattr(self, 'matter_map'):
            # Fallback: SparseMap sin soporte de tensores
            coord = self._coord_to_key(coords)
            # Almacenar magnitud en SparseMap y tensor en dict auxiliar
            magnitude = torch.sum(state_vector.abs().pow(2)).item()
            self.matter_map.insert(coord, magnitude)
            if not hasattr(self, '_state_tensors'):
                self._state_tensors = {}
            key = (coord.x, coord.y, coord.z)
            self._state_tensors[key] = state_vector.to(self.device)
        else:
            # Fallback completo a Python
            self.matter[coords] = state_vector.to(self.device)
        
        self.activate_neighborhood(coords)

    def get_state_at(self, coords):
        """
        Obtiene el estado cuántico en una coordenada.
        Si no hay materia, devuelve el Estado de Vacío (QED).
        """
        if self._use_cpp_native:
            # Recuperar tensor directamente de C++
            coord = self._coord_to_key(coords)
            if self.matter_map.contains_coord(coord):
                tensor = self.matter_map.get_tensor(coord)
                if tensor.numel() > 0:
                    return tensor
        elif hasattr(self, 'matter_map'):
            # Fallback: Recuperar de dict auxiliar
            coord = self._coord_to_key(coords)
            key = (coord.x, coord.y, coord.z)
            if key in self._state_tensors:
                return self._state_tensors[key]
        else:
            # Fallback completo a Python
            if coords in self.matter:
                return self.matter[coords]
        
        # Si no hay materia, devolver el estado del vacío
        return self.vacuum.get_fluctuation(coords, self.step_count)

    def activate_neighborhood(self, coords, radius=1):
        """Marca el vecindario espacial para ser procesado."""
        x, y = coords[0], coords[1]
        # Soportamos 2D y 3D dinámicamente
        z = coords[2] if len(coords) > 2 else None
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if z is not None:
                    for dz in range(-radius, radius + 1):
                        self.active_region.add((x+dx, y+dy, z+dz))
                else:
                    self.active_region.add((x+dx, y+dy))

    def step(self):
        """
        Avanza la simulación. 
        Solo procesa la 'Región Activa', pero teniendo en cuenta el vacío circundante.
        """
        if self._use_cpp_native:
            # Versión optimizada con tensores nativos
            # Usar una nueva instancia de SparseMap para el siguiente estado
            next_matter_map = atheria_core.SparseMap()
        elif hasattr(self, 'matter_map'):
            next_state_tensors = {}
        else:
            next_matter = {}
        
        next_active_region = set()
        self.step_count += 1
        
        # 1. Construir tensores locales para las regiones activas
        processed_coords = list(self.active_region)
        
        for coord in processed_coords:
            # Recolectar vecindario 3x3 (densificación local)
            # Pasamos el vecindario por la U-Net
            # Obtenemos nuevo estado central
            
            # (Simulación simplificada de la inferencia)
            current_state = self.get_state_at(coord)
            
            # Si la energía es alta, la conservamos
            energy = torch.sum(current_state.abs().pow(2))
            if energy > 0.01: # Umbral de existencia
                if self._use_cpp_native:
                    # Almacenar directamente en C++ (tensor nativo)
                    coord_key = self._coord_to_key(coord)
                    # Aquí usaríamos el modelo para obtener el nuevo estado
                    # Por ahora, conservamos el estado actual
                    next_matter_map.insert_tensor(coord_key, current_state)
                elif hasattr(self, 'matter_map'):
                    coord_key = self._coord_to_key(coord)
                    key = (coord_key.x, coord_key.y, coord_key.z)
                    next_state_tensors[key] = current_state
                else:
                    next_matter[coord] = current_state
                
                # Reactivar vecinos para el siguiente frame
                self._activate_neighborhood_for_next(coord, next_active_region)
        
        # Actualizar estado
        if self._use_cpp_native:
            self.matter_map = next_matter_map
            return self.matter_map.size()
        elif hasattr(self, 'matter_map'):
            # Reconstruir SparseMap y diccionario auxiliar
            self.matter_map.clear()
            if hasattr(self, '_state_tensors'):
                self._state_tensors.clear()
            self._state_tensors = {}
            for key, tensor in next_state_tensors.items():
                coord = atheria_core.Coord3D(key[0], key[1], key[2])
                magnitude = torch.sum(tensor.abs().pow(2)).item()
                self.matter_map.insert(coord, magnitude)
                self._state_tensors[key] = tensor
            return len(self._state_tensors)
        else:
            self.matter = next_matter
            self.active_region = next_active_region
            return len(self.matter)
    
    def _activate_neighborhood_for_next(self, coords, next_active_region, radius=1):
        """Helper para activar vecindario en el siguiente frame"""
        x, y = coords[0], coords[1]
        z = coords[2] if len(coords) > 2 else None
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if z is not None:
                    for dz in range(-radius, radius + 1):
                        next_active_region.add((x+dx, y+dy, z+dz))
                else:
                    next_active_region.add((x+dx, y+dy))
    
    def get_matter_count(self):
        """Retorna el número de partículas de materia almacenadas"""
        if self._use_cpp_native:
            return self.matter_map.size()
        elif hasattr(self, 'matter_map'):
            return len(self._state_tensors) if hasattr(self, '_state_tensors') else 0
        else:
            return len(self.matter)
    
    def clear(self):
        """Limpia toda la materia del universo"""
        if self._use_cpp_native:
            self.matter_map.clear()
        elif hasattr(self, 'matter_map'):
            self.matter_map.clear()
            if hasattr(self, '_state_tensors'):
                self._state_tensors.clear()
        else:
            self.matter.clear()
        self.active_region.clear()
    
    def get_storage_info(self):
        """Retorna información sobre el método de almacenamiento"""
        if self._use_cpp_native:
            return {
                'method': 'C++ Native Tensors',
                'cpp_available': True,
                'torch_support': True,
                'auxiliary_dict': False
            }
        elif hasattr(self, 'matter_map'):
            return {
                'method': 'C++ SparseMap (auxiliary dict)',
                'cpp_available': True,
                'torch_support': False,
                'auxiliary_dict': True
            }
        else:
            return {
                'method': 'Python Dict',
                'cpp_available': False,
                'torch_support': False,
                'auxiliary_dict': False
            }

    def compile_model(self):
        """
        Compila el modelo para optimización (no-op para SparseQuantumEngineCppV2).
        """
        pass

    def get_model_for_params(self):
        """
        Retorna el modelo para contar parámetros.
        """
        return self.model

    def get_initial_state(self, batch_size=1):
        """
        Retorna el estado inicial (dummy para SparseQuantumEngineCppV2).
        """
        # SparseQuantumEngineCppV2 maneja su propio estado interno (sparse map)
        # Retornamos un tensor dummy para satisfacer la API del trainer
        return torch.zeros(batch_size, self.d_state, 64, 64, device=self.device)

    def evolve_step(self, current_psi):
        """
        Evoluciona el estado un paso.
        """
        self.step()
        return current_psi

