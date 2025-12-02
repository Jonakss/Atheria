import torch
import numpy as np
import math
import logging

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
    def __init__(self, model, d_state, device='cpu', grid_size=256):
        self.model = model
        self.d_state = d_state
        self.device = device
        self.grid_size = grid_size
        self.vacuum = HarmonicVacuum(d_state, device)
        
        # La Materia: Diccionario {(x,y,z): Tensor}
        self.matter = {} 
        self.active_coords = set()
        self.step_count = 0

    def initialize_matter(self, mode='random', strength=1.0):
        """
        Inicializa la materia del universo.
        Soporta 'ionq' para Quantum Genesis.
        """
        from .qca_engine import QuantumState
        
        logging.info(f"🌌 HarmonicEngine: Initializing matter with mode='{mode}'...")
        
        # Usamos QuantumState para generar la distribución inicial (sea random o ionq)
        # Esto reutiliza la lógica de conexión a IonQ de qca_engine
        qs = QuantumState(self.grid_size, self.d_state, self.device, initial_mode=mode)
        
        # Si el modo es ionq o complex_noise, qs.psi tendrá datos
        # Convertimos ese estado denso a partículas en el HarmonicEngine
        if qs.psi is not None:
            # Extraer energía para decidir dónde poner partículas
            # qs.psi es [1, H, W, d_state]
            psi = qs.psi[0]
            density = psi.abs().pow(2).sum(dim=-1) # [H, W]
            
            # Umbral para crear materia
            # En modo ionq/noise, queremos que surjan partículas en los picos
            threshold = density.mean() + density.std()
            
            indices = torch.nonzero(density > threshold)
            
            count = 0
            for idx in indices:
                y, x = idx[0].item(), idx[1].item()
                # Tomamos el estado del tensor y lo inyectamos como materia
                state_vec = psi[y, x] * strength
                self.add_matter(x, y, 0, state_vec) # Z=0 por defecto
                count += 1
                
            logging.info(f"✨ Quantum Genesis: {count} particles created from {mode} distribution.")
        else:
            logging.warning("⚠️ QuantumState returned None. No matter initialized.")

    def get_dense_state(self, check_pause_callback=None):
        """
        Retorna el estado denso completo del universo (o viewport).
        Compatible con la interfaz de CartesianEngine.
        """
        # Usar centro del grid (0,0,0) y tamaño completo
        # Nota: En universo infinito, esto solo muestra la región central
        return self.get_viewport_tensor((0, 0, 0), self.grid_size, self.step_count * 0.1)

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
        
        # 1. Identificar Chunks Activos (agrupar coordenadas en bloques de 16x16x16)
        CHUNK_SIZE = 16
        active_chunks = set()
        
        for (x, y, z) in self.active_coords:
            cx, cy, cz = x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE
            active_chunks.add((cx, cy, cz))
            
            # Activar vecinos también para permitir propagación
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    active_chunks.add((cx + dx, cy + dy, cz))
        
        next_matter = {}
        next_active_coords = set()
        
        # 2. Procesar cada Chunk
        for (cx, cy, cz) in active_chunks:
            # Coordenadas base del chunk
            base_x, base_y, base_z = cx * CHUNK_SIZE, cy * CHUNK_SIZE, cz * CHUNK_SIZE
            
            # Crear grid denso para el chunk (con padding para contexto)
            PADDING = 2 # Depende del kernel del modelo
            grid_size = CHUNK_SIZE + 2 * PADDING
            
            # Tensor local: [1, C, D, H, W] o [1, C, H, W] si es 2D slice-based
            # Asumimos modelo 2D slice-based por ahora (z-layer) para simplificar
            # O si es 3D, necesitamos un tensor 5D. 
            # Por simplicidad y compatibilidad con U-Net 2D, procesamos slice central Z
            
            # Generar coordenadas para el grid local
            xs = torch.arange(base_x - PADDING, base_x + CHUNK_SIZE + PADDING, device=self.device)
            ys = torch.arange(base_y - PADDING, base_y + CHUNK_SIZE + PADDING, device=self.device)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
            
            # Aplanar
            coords_flat = torch.stack([
                grid_x.flatten(),
                grid_y.flatten(),
                torch.full_like(grid_x.flatten(), base_z) # Procesamos plano Z del chunk
            ], dim=1).float()
            
            # Obtener Vacío
            local_state = self.vacuum.get_state(coords_flat, self.step_count * 0.1)
            local_state = local_state.view(1, grid_size, grid_size, self.d_state)
            local_state = local_state.permute(0, 3, 1, 2) # [1, C, H, W]
            
            # Superponer Materia Existente
            # (Optimización: Solo iterar sobre materia conocida en este rango)
            # Por ahora iteramos todo (lento, pero funcional para prototipo)
            # TODO: Usar Spatial Hash para búsqueda rápida
            for (mx, my, mz), m_state in self.matter.items():
                if mz == base_z and \
                   base_x - PADDING <= mx < base_x + CHUNK_SIZE + PADDING and \
                   base_y - PADDING <= my < base_y + CHUNK_SIZE + PADDING:
                    
                    lx = int(mx - (base_x - PADDING))
                    ly = int(my - (base_y - PADDING))
                    local_state[0, :, ly, lx] = m_state
            
            # 3. Inferencia
            with torch.no_grad():
                # Asumimos que el modelo retorna delta o nuevo estado
                # Entrada: [1, C, H, W] -> Salida: [1, C, H, W]
                output = self.model(local_state)
                
            # 4. Actualizar Materia (Dispersión)
            # Extraer solo la región central (sin padding)
            center_output = output[0, :, PADDING:-PADDING, PADDING:-PADDING] # [C, 16, 16]
            
            # Thresholding para persistencia
            # Si la energía > umbral, guardamos la partícula
            energy = center_output.pow(2).sum(dim=0) # [16, 16]
            active_indices = torch.nonzero(energy > 0.01) # Umbral de existencia
            
            for idx in active_indices:
                ly, lx = idx[0].item(), idx[1].item()
                gx, gy = base_x + lx, base_y + ly
                
                new_state = center_output[:, ly, lx]
                next_matter[(gx, gy, base_z)] = new_state
                next_active_coords.add((gx, gy, base_z))
                
        self.matter = next_matter
        self.active_coords = next_active_coords
        
        return len(self.matter)

    def compile_model(self):
        """
        Compila el modelo para optimización (no-op para HarmonicEngine).
        """
        pass

    def get_model_for_params(self):
        """
        Retorna el modelo para contar parámetros.
        """
        return self.model

    def get_initial_state(self, batch_size=1):
        """
        Retorna el estado inicial (dummy para HarmonicEngine).
        """
        # HarmonicEngine maneja su propio estado interno (self.matter)
        # Retornamos un tensor dummy para satisfacer la API del trainer
        return torch.zeros(batch_size, self.d_state, self.grid_size, self.grid_size, device=self.device)

    def evolve_step(self, current_psi):
        """
        Evoluciona el estado un paso.
        """
        self.step()
        return current_psi