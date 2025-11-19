import torch
import numpy as np
import math

class HarmonicVacuum:
    """
    Generador de Vacío basado en Interferencias de Ondas (QFT Procedural).
    Reemplaza al hash aleatorio por un campo continuo de ondas estacionarias.
    Esto crea un 'terreno' dinámico sobre el cual la materia puede interactuar.
    """
    def __init__(self, d_state, device, complexity=5):
        self.d_state = d_state
        self.device = device
        self.complexity = complexity
        
        # --- FIRMA DEL UNIVERSO ---
        # Generamos vectores de onda (k) y frecuencias (w) aleatorias pero fijas.
        # Esto define las "Leyes del Vacío" de esta simulación específica.
        
        # k_vecs: Dirección y frecuencia espacial de las ondas.
        # Shape: [d_state, complexity, 3] -> (Canales, Ondas por canal, Dimensiones XYZ)
        # Escala 0.3 para ondas suaves y largas.
        self.k_vecs = torch.randn(d_state, complexity, 3, device=device) * 0.3
        
        # omegas: Frecuencia temporal (velocidad de oscilación).
        self.omegas = torch.randn(d_state, complexity, device=device) * 0.1
        
        # phases: Fase inicial aleatoria.
        self.phases = torch.rand(d_state, complexity, device=device) * 2 * math.pi

    def get_state(self, coords_tensor, t):
        """
        Calcula el estado del vacío en un conjunto de coordenadas (x,y,z) al tiempo t.
        Matemática: Sum( A * sin(k*r - w*t + phi) )
        """
        # Asegurar que t sea tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
            
        # Preparar dimensiones para broadcasting
        # coords: [N, 3] -> [N, 1, 1, 3]
        r = coords_tensor.unsqueeze(1).unsqueeze(1) 
        
        # k: [1, D, C, 3]
        k = self.k_vecs.unsqueeze(0)
        
        # Producto punto k*r (Fase Espacial)
        # Sumamos en la última dimensión (xyz)
        k_r = torch.sum(r * k, dim=-1) # -> [N, D, C]
        
        # Componente Temporal w*t
        w_t = self.omegas.unsqueeze(0) * t # -> [1, D, C]
        
        # Fase total
        phi = self.phases.unsqueeze(0)
        
        # La Onda: sin(k*r - w*t + phi)
        wave = torch.sin(k_r - w_t + phi)
        
        # Interferencia: Sumar todas las ondas de complejidad C
        field_val = torch.sum(wave, dim=2) # -> [N, D]
        
        # Normalización de Energía de Vacío (QED Zero Point Energy)
        # El vacío debe ser débil (0.05) para no eclipsar a la materia real (1.0)
        return field_val * 0.05

class SparseHarmonicEngine:
    """
    Motor de Inferencia Masiva.
    Combina Materia Real (almacenada en diccionario) con Vacío Armónico (calculado).
    """
    def __init__(self, model, d_state, device='cpu'):
        self.model = model
        self.d_state = d_state
        self.device = device
        self.vacuum = HarmonicVacuum(d_state, device)
        
        # La Materia: Diccionario {(x,y,z): Tensor}
        self.matter = {} 
        self.active_coords = set()
        self.step_count = 0

    def add_matter(self, x, y, z, state):
        """Inyecta materia real en el universo."""
        self.matter[(x,y,z)] = state.to(self.device)
        self.active_coords.add((x,y,z))

    def get_viewport_tensor(self, center, size_xy, t):
        """
        Genera un tensor denso [H, W, C] para visualizar una región.
        Combina la materia guardada + el vacío generado.
        Ideal para enviar al HolographicViewer.
        """
        cx, cy, cz = center
        half = size_xy // 2
        
        # 1. Generar coordenadas del viewport
        xs = torch.arange(cx - half, cx + half, device=self.device)
        ys = torch.arange(cy - half, cy + half, device=self.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        
        # Aplanar para batch processing
        coords_flat = torch.stack([
            grid_x.flatten(), 
            grid_y.flatten(), 
            torch.full_like(grid_x.flatten(), cz)
        ], dim=1).float()
        
        # 2. Calcular Vacío de Fondo (Base)
        viewport_state = self.vacuum.get_state(coords_flat, t)
        viewport_state = viewport_state.view(size_xy, size_xy, self.d_state)
        
        # 3. Superponer Materia Real
        # (Esto es lento en Python puro, se optimiza con tensores dispersos en producción)
        # Aquí hacemos una implementación simple
        local_matter_mask = torch.zeros(size_xy, size_xy, 1, device=self.device)
        
        for (mx, my, mz), m_state in self.matter.items():
            # Si la materia está dentro del viewport y en el plano Z correcto
            if mz == cz and abs(mx - cx) < half and abs(my - cy) < half:
                lx = int(mx - (cx - half))
                ly = int(my - (cy - half))
                viewport_state[ly, lx] = m_state
                local_matter_mask[ly, lx] = 1.0
                
        return viewport_state

    def step(self):
        self.step_count += 1
        # Lógica de actualización física (Chunk-based inference) pendiente
        # ...
        return len(self.matter)