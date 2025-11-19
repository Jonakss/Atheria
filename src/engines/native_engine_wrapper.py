"""
Wrapper para integrar el motor nativo de C++ (atheria_core.Engine) con el frontend.

Este wrapper proporciona una interfaz compatible con Aetheria_Motor para que
el motor nativo de alto rendimiento pueda usarse como reemplazo directo.
"""
import torch
import logging
import numpy as np
from typing import Optional

try:
    import atheria_core
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    logging.warning("atheria_core no disponible. El motor nativo no funcionará.")

from ..engines.qca_engine import QuantumState


class NativeEngineWrapper:
    """
    Wrapper que envuelve atheria_core.Engine para mantener compatibilidad
    con el código existente que usa Aetheria_Motor.
    
    Convierte entre el formato disperso del motor nativo y el formato
    denso (grid) usado por el frontend.
    """
    
    def __init__(self, grid_size: int, d_state: int, device: str = "cpu", cfg=None):
        """
        Inicializa el wrapper del motor nativo.
        
        Args:
            grid_size: Tamaño del grid para visualización (debe coincidir con inference_grid_size)
            d_state: Dimensión del estado cuántico
            device: Dispositivo ('cpu' o 'cuda')
            cfg: Configuración del experimento (opcional)
        """
        if not NATIVE_AVAILABLE:
            raise ImportError("atheria_core no está disponible. Instala el módulo C++ primero.")
        
        self.grid_size = int(grid_size)
        self.d_state = int(d_state)
        self.device_str = device
        self.device = torch.device(device)
        self.cfg = cfg
        
        # Inicializar motor nativo
        self.native_engine = atheria_core.Engine(d_state=d_state, device=device)
        
        # Estado cuántico para compatibilidad (denso)
        # El motor nativo usa formato disperso, pero necesitamos denso para visualización
        self.state = QuantumState(grid_size, d_state, device)
        
        # Configuración
        self.grid_size = grid_size
        self.d_state = d_state
        
        # Estado interno
        self.model_loaded = False
        self.step_count = 0
        
        # Para compatibilidad con código existente
        self.operator = None  # El modelo está cargado en el motor nativo
        self.original_model = None
        self.optimizer = None
        self.has_memory = False
        self.is_compiled = False
        
        # Para visualizaciones (delta_psi)
        self.last_delta_psi = None
        self.last_psi_input = None
        self.last_delta_psi_decay = None
        
        logging.info(f"NativeEngineWrapper inicializado (grid_size={grid_size}, d_state={d_state}, device={device})")
    
    def load_model(self, model_path: str) -> bool:
        """
        Carga un modelo TorchScript (.pt) en el motor nativo.
        
        Args:
            model_path: Ruta al archivo .pt de TorchScript
            
        Returns:
            True si se cargó exitosamente
        """
        success = self.native_engine.load_model(model_path)
        self.model_loaded = success
        if success:
            logging.info(f"✅ Modelo TorchScript cargado en motor nativo: {model_path}")
        else:
            logging.error(f"❌ Error al cargar modelo TorchScript: {model_path}")
        return success
    
    def evolve_internal_state(self):
        """Evoluciona el estado interno usando el motor nativo."""
        if not self.model_loaded:
            logging.warning("Modelo no cargado. No se puede evolucionar el estado.")
            return
        
        # Ejecutar paso nativo (todo en C++)
        particle_count = self.native_engine.step_native()
        self.step_count += 1
        
        # Convertir estado disperso a denso para visualización
        # Esto es necesario porque el frontend espera un grid denso
        self._update_dense_state_from_sparse()
    
    def _update_dense_state_from_sparse(self):
        """
        Convierte el estado disperso del motor nativo a formato denso (grid)
        para compatibilidad con el frontend.
        
        El motor nativo almacena partículas dispersas, pero el frontend necesita
        un grid denso. Iteramos sobre todo el grid y obtenemos el estado desde
        el motor nativo (que genera vacío automáticamente si no hay partícula).
        """
        # Inicializar grid denso
        if self.state.psi is None:
            self.state.psi = torch.zeros(
                1, self.grid_size, self.grid_size, self.d_state,
                dtype=torch.complex64, device=self.device
            )
        
        # Iterar sobre todo el grid y obtener estado desde motor nativo
        # El motor nativo genera vacío automáticamente con HarmonicVacuum
        try:
            # Para optimizar, podríamos iterar solo sobre coordenadas activas
            # pero por simplicidad, iteramos sobre todo el grid
            # (puede optimizarse después si es necesario)
            BATCH_SIZE = 100  # Procesar en batches para mejor rendimiento
            
            coords_list = []
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    coords_list.append(atheria_core.Coord3D(x, y, 0))
            
            # Obtener estados en batch
            for i in range(0, len(coords_list), BATCH_SIZE):
                batch_coords = coords_list[i:i+BATCH_SIZE]
                for coord in batch_coords:
                    try:
                        state_tensor = self.native_engine.get_state_at(coord)
                        
                        # Si el tensor tiene la forma correcta, copiarlo al grid denso
                        if state_tensor.shape == (self.d_state,):
                            # Mover a dispositivo correcto
                            if state_tensor.is_cuda and self.device.type == 'cpu':
                                state_tensor = state_tensor.cpu()
                            elif not state_tensor.is_cuda and self.device.type == 'cuda':
                                state_tensor = state_tensor.cuda()
                            
                            # Copiar al estado denso (batch, H, W, d_state)
                            self.state.psi[0, coord.y, coord.x] = state_tensor
                    except Exception as e:
                        logging.debug(f"Error obteniendo estado en ({coord.x}, {coord.y}): {e}")
                        
        except Exception as e:
            logging.warning(f"Error convirtiendo estado disperso a denso: {e}")
            # En caso de error, mantener grid vacío (ya inicializado)
    
    def get_model_for_params(self):
        """Retorna el modelo para acceso a parámetros (compatibilidad)."""
        return self.original_model if self.original_model else self
    
    def compile_model(self):
        """Método de compatibilidad - el modelo nativo ya está compilado."""
        self.is_compiled = True
        logging.info("Modelo nativo: ya está optimizado (compilado en C++)")
    
    def add_initial_particles(self, num_particles: int = 10):
        """
        Agrega partículas iniciales aleatorias al motor nativo.
        
        Args:
            num_particles: Número de partículas a agregar
        """
        # Generar partículas aleatorias en el grid
        for _ in range(num_particles):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            z = 0  # Para 2D, z=0
            
            # Estado inicial pequeño (energía baja)
            initial_state = torch.randn(self.d_state, dtype=torch.complex64) * 0.1
            
            # Agregar al motor nativo
            coord = atheria_core.Coord3D(x, y, z)
            self.native_engine.add_particle(coord, initial_state)
        
        logging.info(f"✅ {num_particles} partículas iniciales agregadas al motor nativo")
        
        # Actualizar estado denso
        self._update_dense_state_from_sparse()

