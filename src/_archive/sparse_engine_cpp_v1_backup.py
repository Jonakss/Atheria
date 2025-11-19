# src/engines/sparse_engine_cpp.py
"""
Motor de simulación disperso usando el núcleo C++ atheria_core.

Esta es una versión mejorada del SparseQuantumEngine que usa SparseMap de C++
en lugar de diccionarios Python para mayor rendimiento.
"""
import torch
import numpy as np
import math

try:
    import atheria_core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
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


class SparseQuantumEngineCpp:
    """
    Motor de simulación para un universo potencialmente infinito usando C++.
    Maneja la dualidad Materia (almacenada) vs Vacío (generado).
    
    Esta versión usa atheria_core.SparseMap para almacenar materia, lo que proporciona
    mejor rendimiento que diccionarios Python puros.
    """
    def __init__(self, model, d_state, device='cpu', vacuum_mode='active'):
        self.model = model
        self.d_state = d_state
        self.device = device
        self.vacuum = QuantumVacuum(d_state, device)
        
        # El Universo: SparseMap C++ {(x, y, z) -> estado}
        # Usamos un hash de coordenadas como clave
        if CPP_AVAILABLE:
            self.matter_map = atheria_core.SparseMap()
            self._use_cpp = True
        else:
            # Fallback a Python si C++ no está disponible
            self.matter = {}
            self._use_cpp = False
        
        # Conjunto de celdas que necesitan actualización en el siguiente paso
        # (Incluye materia y su vecindario inmediato)
        self.active_region = set()
        
        self.step_count = 0
    
    def _coord_to_key(self, coords):
        """Convierte coordenadas (x, y, z) a una clave numérica para SparseMap"""
        x, y = coords[0], coords[1]
        z = coords[2] if len(coords) > 2 else 0
        # Codificar coordenadas en una clave única
        # Usamos 20 bits para x, 20 bits para y, 24 bits para z
        return (x << 44) | (y << 24) | z
    
    def _key_to_coord(self, key):
        """Convierte una clave numérica de SparseMap a coordenadas (x, y, z)"""
        z = key & 0xFFFFFF  # Últimos 24 bits
        y = (key >> 24) & 0xFFFFF  # Siguientes 20 bits
        x = (key >> 44) & 0xFFFFF  # Primeros 20 bits
        return (x, y, z)
    
    def _store_state(self, coords, state_vector):
        """Almacena un estado en el mapa (compatible con C++ y Python)"""
        if self._use_cpp:
            key = self._coord_to_key(coords)
            # Para almacenar tensores complejos, necesitamos serializar
            # Por ahora, almacenamos solo la magnitud como placeholder
            # TODO: Implementar serialización completa de tensores
            magnitude = torch.sum(state_vector.abs().pow(2)).item()
            self.matter_map[key] = magnitude
            # Almacenar referencia al tensor en un diccionario auxiliar
            if not hasattr(self, '_state_tensors'):
                self._state_tensors = {}
            self._state_tensors[key] = state_vector.to(self.device)
        else:
            self.matter[coords] = state_vector.to(self.device)
    
    def _get_state(self, coords):
        """Obtiene un estado del mapa (compatible con C++ y Python)"""
        if self._use_cpp:
            key = self._coord_to_key(coords)
            if key in self.matter_map:
                # Recuperar tensor del diccionario auxiliar
                return self._state_tensors.get(key, None)
            return None
        else:
            return self.matter.get(coords, None)

    def add_particle(self, coords, state_vector):
        """Inyecta una excitación (partícula) en el campo."""
        self._store_state(coords, state_vector)
        self.activate_neighborhood(coords)

    def get_state_at(self, coords):
        """
        Obtiene el estado cuántico en una coordenada.
        Si no hay materia, devuelve el Estado de Vacío (QED).
        """
        state = self._get_state(coords)
        if state is not None:
            return state
        else:
            # Aquí está la magia: El vacío devuelve energía, no ceros.
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
        if self._use_cpp:
            # Limpiar diccionario auxiliar al inicio del step
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
                if self._use_cpp:
                    key = self._coord_to_key(coord)
                    # Almacenar en el siguiente estado
                    magnitude = torch.sum(current_state.abs().pow(2)).item()
                    # Aquí usaríamos el modelo para obtener el nuevo estado
                    # Por ahora, conservamos el estado actual
                    next_state_tensors[key] = current_state
                else:
                    next_matter[coord] = current_state
                
                # Reactivar vecinos para el siguiente frame
                self._activate_neighborhood_for_next(coord, next_active_region)
        
        # Actualizar estado
        if self._use_cpp:
            # Limpiar mapa y reconstruir
            self.matter_map.clear()
            self._state_tensors = {}
            for key, tensor in next_state_tensors.items():
                magnitude = torch.sum(tensor.abs().pow(2)).item()
                self.matter_map[key] = magnitude
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
        if self._use_cpp:
            return len(self.matter_map)
        else:
            return len(self.matter)
    
    def clear(self):
        """Limpia toda la materia del universo"""
        if self._use_cpp:
            self.matter_map.clear()
            if hasattr(self, '_state_tensors'):
                self._state_tensors.clear()
        else:
            self.matter.clear()
        self.active_region.clear()

