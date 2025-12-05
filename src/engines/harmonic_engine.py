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

    def regenerate(self):
        """
        Regenera los parámetros del vacío (k, w, phi) para crear un nuevo "universo".
        Esto cambia el patrón de interferencia de fondo.
        """
        self.k_vecs = torch.randn(self.d_state, self.complexity, 3, device=self.device) * 0.3
        self.omegas = torch.randn(self.d_state, self.complexity, device=self.device) * 0.1
        self.phases = torch.rand(self.d_state, self.complexity, device=self.device) * 2 * math.pi

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

class DummyState:
    """Clase ligera para envolver el estado denso."""
    def __init__(self, psi):
        self.psi = psi

class SparseHarmonicEngine:
    """
    Motor de Inferencia Masiva.
    Combina Materia Real (almacenada en diccionario) con Vacío Armónico (calculado).
    """
    def __init__(self, model, d_state, device='cpu', grid_size=256):
        self.model = model
        self.operator = model # Alias for compatibility with Trainer
        self.d_state = d_state
        self.device = device
        self.grid_size = grid_size
        self.vacuum = HarmonicVacuum(d_state, device)
        
        # La Materia: Diccionario {(x,y,z): Tensor}
        self.matter = {} 
        self.active_coords = set()
        self.step_count = 0

        # Cache para optimización de visualización
        self._cached_state_obj = None
        self._cache_step = -1
        self._cache_valid = False

        # Quantum Tools
        from ..physics.quantum_collapse import IonQCollapse
        from ..physics.steering import QuantumSteering
        self.collider = IonQCollapse(device)
        self.steering = QuantumSteering(device)

    @property
    def state(self):
        """
        Interfaz dummy para compatibilidad con handlers.
        Retorna un objeto con atributo .psi que contiene el estado denso, cacheado si es posible.
        """
        # Verificar si el caché es válido para el paso actual
        if self._cache_valid and self._cached_state_obj is not None and self._cache_step == self.step_count:
            return self._cached_state_obj

        # Generar estado denso on-demand
        psi_dense = self.get_viewport_tensor((0, 0, 0), self.grid_size, self.step_count * 0.1)
        self._cached_state_obj = DummyState(psi_dense)
        self._cache_step = self.step_count
        self._cache_valid = True

        return self._cached_state_obj

    @state.setter
    def state(self, new_state):
        """
        Setter mágico: Cuando el servidor asigna motor.state = QuantumState(...),
        interceptamos para reiniciar el motor Armónico correctamente.
        """
        logging.info("🌌 HarmonicEngine: Intercepting state reset. Regenerating universe...")

        # 1. Regenerar el Vacío (Nuevas leyes de la física para este universo)
        self.vacuum.regenerate()

        # 2. Resetear contadores
        self.step_count = 0
        self.matter = {}
        self.active_coords = set()

        # Invalidar caché
        self._cache_valid = False

        # 3. Inyectar materia del nuevo estado
        if hasattr(new_state, 'psi') and new_state.psi is not None:
            self._ingest_dense_state(new_state.psi)

    def _ingest_dense_state(self, psi_tensor, strength=1.0):
        """Convierte un tensor denso [1, H, W, C] en partículas dispersas."""
        # Invalidar caché al cambiar materia
        self._cache_valid = False

        psi = psi_tensor[0]
        density = psi.abs().pow(2).sum(dim=-1) # [H, W]

        # Umbral dinámico
        threshold = density.mean() + density.std()

        indices = torch.nonzero(density > threshold)
        count = 0
        for idx in indices:
            y, x = idx[0].item(), idx[1].item()
            state_vec = psi[y, x] * strength
            self.add_matter(x, y, 0, state_vec)
            count += 1
        logging.info(f"✨ Harmonic Reset: {count} particles initialized from new state.")

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
            self._ingest_dense_state(qs.psi, strength)
        else:
            logging.warning("⚠️ QuantumState returned None. No matter initialized.")

    def get_dense_state(self, roi=None, check_pause_callback=None):
        """
        Retorna el estado denso completo del universo (o viewport).
        Compatible con la interfaz de CartesianEngine.
        """
        # Usar centro del grid (0,0,0) y tamaño completo
        # Nota: En universo infinito, esto solo muestra la región central
        # Si se pasa ROI, podríamos optimizar, pero por ahora ignoramos ROI y devolvemos todo o el viewport central
        return self.get_viewport_tensor((0, 0, 0), self.grid_size, self.step_count * 0.1)

    def add_matter(self, x, y, z, state):
        """Inyecta materia real en el universo."""
        self.matter[(x,y,z)] = state.to(self.device)
        self.active_coords.add((x,y,z))
        # Invalidar caché
        self._cache_valid = False

    def get_viewport_tensor(self, center, size_xy, t):
        """
        Genera un tensor denso [H, W, C] para visualizar una región.
        Combina la materia guardada + el vacío generado.
        Ideal para enviar al HolographicViewer.
        """
        cx, cy, cz = center
        half = size_xy // 2
        
        # 1. Generar coordenadas del viewport
        # Usamos lógica robusta: start + size para garantizar el tamaño exacto
        start_x = cx - half
        start_y = cy - half
        
        # Logging para depuración de tamaños (temporal)
        # logging.debug(f"HarmonicEngine Viewport: size={size_xy}, half={half}, start_x={start_x}")
        
        xs = torch.arange(start_x, start_x + size_xy, device=self.device)
        ys = torch.arange(start_y, start_y + size_xy, device=self.device)
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
                # Fix complex casting warning: m_state might be complex if coming from quantum tools
                # But viewport_state is initialized as real (implied by context, though line 193 makes it complex later)
                # Actually line 168: viewport_state = self.vacuum.get_state(...) which returns real.
                # So we must cast m_state to real if it's complex.
                if m_state.is_complex():
                    viewport_state[ly, lx] = m_state.real # Or .abs() depending on semantics. Real part is standard for scalar field.
                else:
                    viewport_state[ly, lx] = m_state
                local_matter_mask[ly, lx] = 1.0
                
        # Permutar a [1, H, W, C] para compatibilidad con CartesianEngine y VisualizationPipeline
        viewport_state = viewport_state.unsqueeze(0)
        
        return torch.complex(viewport_state, torch.zeros_like(viewport_state))

    def step(self):
        self.step_count += 1
        # El cambio de step_count ya invalida el cache por chequeo de _cache_step,
        # pero para ser explícitos y consistentes:
        self._cache_valid = False
        
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
            
            # 2.5. Optimización: Spatial Hash para búsqueda rápida de materia
            # En lugar de iterar sobre self.matter.items() (lento), buscamos solo en el chunk
            # Asumimos que self.matter está indexado por (x,y,z)
            
            # Iterar sobre el rango del chunk (incluyendo padding)
            # Esto es más rápido si la materia es dispersa pero el chunk es pequeño (20x20)
            # Si el chunk está muy lleno, iterar sobre active_coords y filtrar es mejor
            # Pero dado que active_coords es global, iterar el grid local es O(ChunkSize) constante
            
            # Mejor enfoque híbrido: Iterar sobre active_coords SOLO si están en este chunk
            # Para eso necesitamos un índice espacial. Lo construimos al vuelo o lo mantenemos.
            # Dado que self.matter es un dict, podemos consultar coordenadas directamente.
            
            # Iterar sobre todas las coordenadas del grid local y ver si hay materia
            # Esto es 20x20 = 400 lookups en dict, muy rápido.
            for ly in range(grid_size):
                for lx in range(grid_size):
                    gx = base_x - PADDING + lx
                    gy = base_y - PADDING + ly
                    gz = base_z
                    
                    if (gx, gy, gz) in self.matter:
                        m_state = self.matter[(gx, gy, gz)]
                        
                        # Fix complex casting: Ensure local_state is complex if m_state is complex
                        if m_state.is_complex() and not local_state.is_complex():
                            local_state = local_state.to(torch.complex64)
                            
                        local_state[0, :, ly, lx] = m_state
            
            # 3. Inferencia
            with torch.no_grad():
                # Asumimos que el modelo retorna delta o nuevo estado
                # Entrada: [1, C, H, W] -> Salida: [1, C, H, W]
                if self.model is not None:
                    output = self.model(local_state)
                else:
                    # Si no hay modelo, la materia simplemente decae o se mueve por inercia (simplificado)
                    # O simplemente retornamos el estado local (identidad)
                    output = local_state * 0.99 # Ligero decaimiento para evitar explosión
                
            # 4. Actualizar Materia (Dispersión)
            # Extraer solo la región central (sin padding)
            center_output = output[0, :, PADDING:-PADDING, PADDING:-PADDING] # [C, 16, 16]
            
            # Thresholding para persistencia
            # Si la energía > umbral, guardamos la partícula
            # NOTA: center_output puede ser complejo, usar .abs() para obtener magnitud
            energy = center_output.abs().pow(2).sum(dim=0) # [16, 16] - ahora es real
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

    def evolve_internal_state(self, step=None):
        """
        Alias para step() para compatibilidad.
        """
        self.step()

    def get_visualization_data(self, viz_type: str = "density"):
        """
        Retorna datos de visualización para shaders.
        
        Args:
            viz_type: Tipo de visualización ("density", "phase", "energy", etc.)
            
        Returns:
            dict con 'data' (array 2D) y 'metadata'
        """
        try:
            # Obtener estado denso del viewport
            # [1, H, W, C] complex
            psi = self.get_dense_state()
            
            if psi is None:
                return {"data": None, "type": viz_type, "error": "No state"}
            
            # Reducir canales si múltiples (promediar)
            # psi is [1, H, W, C]
            psi = psi[0]  # [H, W, C]
            
            if viz_type == "density" or viz_type == "magnitude":
                # Densidad = suma de magnitudes al cuadrado sobre canales
                data = psi.abs().pow(2).sum(dim=-1)  # [H, W]
                
            elif viz_type == "phase":
                # Fase del campo agregado (primer canal o promedio)
                phase = torch.angle(psi[..., 0] if psi.shape[-1] > 0 else psi.sum(dim=-1))
                # Normalizar a [0, 1]
                data = (phase / (2 * 3.14159) + 0.5) % 1.0
                
            elif viz_type == "energy":
                data = psi.abs().pow(2).sum(dim=-1)
                
            elif viz_type == "gradient":
                # Gradiente de la densidad
                density = psi.abs().pow(2).sum(dim=-1)
                gy = torch.diff(density, dim=0, prepend=density[:1, :])
                gx = torch.diff(density, dim=1, prepend=density[:, :1])
                data = torch.sqrt(gx**2 + gy**2)
                
            elif viz_type == "real":
                data = psi.real.sum(dim=-1)
                
            elif viz_type == "imag":
                data = psi.imag.sum(dim=-1)
                
            else:
                logging.warning(f"⚠️ Tipo de visualización '{viz_type}' no soportado. Usando density.")
                data = psi.abs().pow(2).sum(dim=-1)
            
            # Convertir a numpy
            data_np = data.cpu().numpy().astype(np.float32)
            
            # NORMALIZACIÓN: Los shaders esperan datos en [0, 1]
            min_val = float(data_np.min())
            max_val = float(data_np.max())
            if max_val > min_val and abs(max_val - min_val) > 1e-10:
                data_np = (data_np - min_val) / (max_val - min_val)
            else:
                # Si todos los valores son iguales, usar 0.5 (gris medio)
                data_np = np.full_like(data_np, 0.5)
            
            return {
                "data": data_np,
                "type": viz_type,
                "shape": list(data_np.shape),
                "min": 0.0,  # Ya normalizado
                "max": 1.0,  # Ya normalizado
                "engine": "SparseHarmonicEngine"
            }
            
        except Exception as e:
            logging.error(f"Error en get_visualization_data: {e}")
            return {"data": None, "type": viz_type, "error": str(e)}

    def apply_tool(self, action, params):
        """
        Aplica una herramienta cuántica al universo armónico.
        """
        # Invalidar caché al aplicar herramienta
        self._cache_valid = False

        logging.info(f"🛠️ HarmonicEngine aplicando herramienta: {action} | Params: {params}")
        
        try:
            # 1. Preparar región local (Viewport centrado en la acción)
            # Para simplificar, usamos un viewport de tamaño fijo alrededor del evento
            # Si no hay coordenadas (global), usamos el centro del universo
            
            cx = int(params.get('x', 0))
            cy = int(params.get('y', 0))
            cz = 0 # Asumimos plano Z=0 para interacción UI
            
            # Tamaño de la región de efecto
            radius = int(params.get('radius', 10))
            if action == 'collapse':
                # Collapse puede ser más grande
                intensity = float(params.get('intensity', 0.5))
                radius = int(self.grid_size * 0.1)
            
            region_size = radius * 2 + 1
            
            # Obtener estado denso local
            # [1, H, W, C] -> [1, region_size, region_size, d_state]
            # get_viewport_tensor retorna [1, H, W, C]
            local_tensor = self.get_viewport_tensor((cx, cy, cz), region_size, self.step_count * 0.1)
            
            # 2. Aplicar herramienta
            new_local_tensor = None
            
            if action == 'collapse':
                intensity = float(params.get('intensity', 0.5))
                # IonQCollapse espera [1, H, W, C]
                new_local_tensor = self.collider.collapse(local_tensor, region_center=(radius, radius), intensity=intensity)
                
            elif action in ['vortex', 'wave', 'soliton']:
                # QuantumSteering espera [1, H, W, C]
                pattern_type = action
                if action == 'wave': pattern_type = 'superposition' # Map wave to superposition or custom
                
                # Steering inject
                new_local_tensor = self.steering.inject(local_tensor, pattern_type, radius, radius, radius=radius//2)
                
            else:
                logging.warning(f"⚠️ Herramienta no soportada: {action}")
                return False
                
            # 3. Actualizar Materia (Dispersión)
            # Convertir el tensor denso modificado de vuelta a partículas
            if new_local_tensor is not None:
                # Extraer datos
                # [1, H, W, C] -> [H, W, C]
                data = new_local_tensor[0]
                
                # Iterar sobre el tensor local y actualizar self.matter
                # Solo actualizamos si la energía es significativa (para mantener sparsity)
                energy = data.abs().pow(2).sum(dim=-1)
                threshold = 0.01
                
                indices = torch.nonzero(energy > threshold)
                
                # Offset para coordenadas globales
                offset_x = cx - radius
                offset_y = cy - radius
                
                count = 0
                for idx in indices:
                    ly, lx = idx[0].item(), idx[1].item()
                    gx, gy = offset_x + lx, offset_y + ly
                    
                    state_vec = data[ly, lx]
                    self.add_matter(gx, gy, cz, state_vec)
                    count += 1
                    
                logging.info(f"✨ Tool '{action}' applied. Updated {count} particles in region.")
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"❌ Error applying tool '{action}': {e}", exc_info=True)
            return False