import torch
import numpy as np
import math

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

class SparseQuantumEngine:
    """
    Motor de simulación para un universo potencialmente infinito.
    Maneja la dualidad Materia (almacenada) vs Vacío (generado).
    """
    def __init__(self, model, d_state, device='cpu', vacuum_mode='active'):
        self.model = model
        self.d_state = d_state
        self.device = device
        self.vacuum = QuantumVacuum(d_state, device)
        
        # El Universo: Diccionario {(x, y, z): Tensor Estado}
        self.matter = {} 
        
        # Conjunto de celdas que necesitan actualización en el siguiente paso
        # (Incluye materia y su vecindario inmediato)
        self.active_region = set()
        
        self.step_count = 0

    def add_particle(self, coords, state_vector):
        """Inyecta una excitación (partícula) en el campo."""
        self.matter[coords] = state_vector.to(self.device)
        self.activate_neighborhood(coords)

    def get_state_at(self, coords):
        """
        Obtiene el estado cuántico en una coordenada.
        Si no hay materia, devuelve el Estado de Vacío (QED).
        """
        if coords in self.matter:
            return self.matter[coords]
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
        next_matter = {}
        next_active_region = set()
        self.step_count += 1
        
        # 1. Construir tensores locales para las regiones activas
        # (Esta es la parte lenta en Python puro, idealmente se hace en C++/CUDA)
        # Para este prototipo, iteramos (lento pero funcional para pruebas)
        
        processed_coords = list(self.active_region)
        
        # Batch processing sería ideal aquí
        for coord in processed_coords:
            # Recolectar vecindario 3x3 (densificación local)
            # Pasamos el vecindario por la U-Net
            # Obtenemos nuevo estado central
            
            # (Simulación simplificada de la inferencia)
            current_state = self.get_state_at(coord)
            
            # Si la energía es alta, la conservamos
            energy = torch.sum(current_state.abs().pow(2))
            if energy > 0.01: # Umbral de existencia
                next_matter[coord] = current_state # Aquí iría el resultado del modelo
                
                # Reactivar vecinos para el siguiente frame
                # (Copiar lógica de activate_neighborhood para next_active_region)
        
        self.matter = next_matter
        self.active_region = next_active_region # Actualizar regiones activas
        
        return len(self.matter)
    
    def get_matter_count(self):
        """Retorna el número de partículas de materia almacenadas"""
        return len(self.matter)
    
    def clear(self):
        """Limpia toda la materia del universo"""
        self.matter.clear()
        self.active_region.clear()