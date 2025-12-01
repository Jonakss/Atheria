import torch
import logging
import torch.nn as nn
from types import SimpleNamespace

from .engines.qca_engine import Aetheria_Motor, QuantumState
from .engines.qca_engine_polar import PolarEngine, QuantumStatePolar
try:
    from .engines.qca_engine_pennylane import QuantumKernel
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

# Importar otros motores
try:
    from .engines.lattice_engine import LatticeEngine
    LATTICE_AVAILABLE = True
except ImportError:
    LATTICE_AVAILABLE = False

try:
    from .engines.harmonic_engine import SparseHarmonicEngine
    HARMONIC_AVAILABLE = True
except ImportError:
    HARMONIC_AVAILABLE = False

class PolarMotorWrapper(Aetheria_Motor):
    """
    Wrapper para adaptar PolarEngine a la interfaz de Aetheria_Motor.
    """
    def __init__(self, model_operator, grid_size, d_state, device, cfg=None):
        # Inicializar Aetheria_Motor pero sobrescribir lo necesario
        super().__init__(model_operator, grid_size, d_state, device, cfg)
        
        # Asegurar que el operador es PolarEngine
        if not isinstance(model_operator, PolarEngine):
            # Si nos pasan un modelo normal, intentamos envolverlo o usarlo tal cual si es compatible
            pass

    def _evolve_logic(self, psi_in):
        """
        Sobrescribe la lógica de evolución para usar coordenadas polares.
        """
        # 1. Convertir estado cartesiano (psi_in) a Polar
        magnitude = psi_in.abs().permute(0, 3, 1, 2) # (1, d_state, H, W)
        phase = psi_in.angle().permute(0, 3, 1, 2)   # (1, d_state, H, W)
        
        state_polar = QuantumStatePolar(magnitude, phase)
        
        # 2. Evolucionar usando PolarEngine
        new_state_polar = self.operator(state_polar)
        
        # 3. Convertir de vuelta a Cartesiano para el sistema
        real, imag = new_state_polar.to_cartesian()
        
        # Volver a (1, H, W, d_state)
        real = real.permute(0, 2, 3, 1)
        imag = imag.permute(0, 2, 3, 1)
        
        new_psi = torch.complex(real, imag)
        
        # Guardar para visualización (aproximado)
        self.last_psi_input = psi_in
        self.last_delta_psi = new_psi - psi_in 
        
        return new_psi

class HybridMotorWrapper(Aetheria_Motor):
    """
    Wrapper para el motor Híbrido (Quantum).
    """
    def __init__(self, model_operator, grid_size, d_state, device, cfg=None):
        super().__init__(model_operator, grid_size, d_state, device, cfg)
        if not PENNYLANE_AVAILABLE:
            logging.warning("⚠️ PennyLane no está instalado. HybridMotor funcionará en modo simulación limitada o fallará.")

    def _evolve_logic(self, psi_in):
        # Adaptar input para QuantumKernel
        x = psi_in.abs().permute(0, 3, 1, 2) 
        
        # Forward pass
        out = self.operator(x)
        
        delta = out.permute(0, 2, 3, 1)
        
        # Aplicar actualización (simple Euler)
        delta_complex = torch.complex(delta, torch.zeros_like(delta))
        
        new_psi = psi_in + delta_complex * 0.1 
        
        # Normalizar
        norm = torch.sqrt(torch.sum(new_psi.abs().pow(2), dim=-1, keepdim=True))
        new_psi = new_psi / (norm + 1e-9)
        
        self.last_psi_input = psi_in
        self.last_delta_psi = delta_complex
        
        return new_psi

class LatticeMotorWrapper:
    """
    Wrapper para LatticeEngine.
    No hereda de Aetheria_Motor porque LatticeEngine funciona muy diferente.
    Implementa la interfaz requerida por SimulationService y DataProcessingService.
    """
    def __init__(self, engine):
        self.engine = engine
        self.device = engine.device
        # Dummy state object for compatibility if accessed directly, though get_dense_state is preferred
        self.state = SimpleNamespace(psi=None)
        # Initialize state.psi to pass validation in handle_play
        self.state.psi = self.get_dense_state()

    def evolve_internal_state(self):
        """Avanza la simulación un paso."""
        self.engine.step()

    def get_dense_state(self, roi=None, check_pause_callback=None):
        """Retorna el estado para visualización."""
        # LatticeEngine retorna densidad de energía [H, W]
        energy = self.engine.get_visualization_data(viz_type="density")
        
        # Convertir a formato compatible con psi [1, H, W, d_state]
        # Usamos energía como magnitud, fase 0
        # Asumimos d_state=1 o repetimos
        if energy is not None:
            H, W = energy.shape
            # Normalizar para visualización
            energy = energy / (energy.max() + 1e-9)
            
            # Crear tensor complejo [1, H, W, 1]
            real = energy.unsqueeze(0).unsqueeze(-1)
            imag = torch.zeros_like(real)
            psi = torch.complex(real, imag)
            # Update internal state for validation
            if self.state:
                self.state.psi = psi
            return psi
        return None
        
    def cleanup(self):
        pass

class HarmonicMotorWrapper:
    """
    Wrapper para SparseHarmonicEngine.
    """
    def __init__(self, engine):
        self.engine = engine
        self.device = engine.device
        self.state = SimpleNamespace(psi=None)
        # Initialize state.psi to pass validation in handle_play
        self.state.psi = self.get_dense_state()

    def evolve_internal_state(self):
        self.engine.step()

    def get_dense_state(self, roi=None, check_pause_callback=None):
        # SparseHarmonicEngine.get_dense_state retorna [H, W, C] real?
        # Revisando código: retorna viewport_state [H, W, C]
        state = self.engine.get_dense_state()
        
        # Convertir a [1, H, W, C] complejo
        # Asumimos que el estado es "real" (magnitud) o complejo?
        # HarmonicVacuum retorna real.
        if state is not None:
            state = state.unsqueeze(0) # [1, H, W, C]
            real = state
            imag = torch.zeros_like(real)
            psi = torch.complex(real, imag)
            # Update internal state for validation
            if self.state:
                self.state.psi = psi
            return psi
        return None

    def cleanup(self):
        pass

def get_motor(config, device, model=None):
    """
    Factory method para obtener la instancia del motor adecuada.
    """
    print("DEBUG: Executing get_motor")
    # Determinar tipo de motor
    engine_type = 'CARTESIAN'
    if hasattr(config, 'ENGINE_TYPE'):
        engine_type = config.ENGINE_TYPE
    elif isinstance(config, dict) and 'ENGINE_TYPE' in config:
        engine_type = config['ENGINE_TYPE']
        
    logging.info(f"🏭 Motor Factory: Solicitado motor tipo '{engine_type}'")
    
    # Parámetros comunes
    grid_size = getattr(config, 'GRID_SIZE_TRAINING', 64)
    if hasattr(config, 'GRID_SIZE_INFERENCE'): 
         grid_size = config.GRID_SIZE_INFERENCE
         
    # Extraer d_state
    d_state = 2
    model_params = getattr(config, 'MODEL_PARAMS', None)
    if model_params:
        if isinstance(model_params, dict):
            d_state = model_params.get('d_state', 2)
        else:
            d_state = getattr(model_params, 'd_state', 2)
            
    # Instanciar el modelo si no viene dado (solo para Cartesian/Polar/Quantum que usan modelo base)
    # Lattice y Harmonic crean su propio "modelo" interno
    
    if engine_type == 'LATTICE':
        if not LATTICE_AVAILABLE:
            raise ImportError("LatticeEngine no disponible.")
        logging.info("Instanciando LatticeEngine...")
        # LatticeEngine args: grid_size, d_state, device, group, beta
        engine = LatticeEngine(grid_size, d_state, device)
        return LatticeMotorWrapper(engine)
        
    elif engine_type == 'HARMONIC':
        if not HARMONIC_AVAILABLE:
            raise ImportError("SparseHarmonicEngine no disponible.")
        logging.info("Instanciando SparseHarmonicEngine...")
        # SparseHarmonicEngine args: model, d_state, device, grid_size
        # Necesita un modelo base? El código dice self.model = model.
        # Si model es None, pasamos None o un dummy?
        engine = SparseHarmonicEngine(model, d_state, device, grid_size)
        return HarmonicMotorWrapper(engine)

    # Para los otros, necesitamos modelo base
    if model is None:
        from .model_loader import instantiate_model
        try:
            model = instantiate_model(config)
        except Exception as e:
            logging.error(f"Error instanciando modelo base: {e}")
            raise

    # Seleccionar Wrapper
    if engine_type == 'POLAR':
        if not isinstance(model, PolarEngine):
            logging.info("Adaptando modelo estándar a PolarEngine...")
            model = PolarEngine(model, grid_size=grid_size)
        return PolarMotorWrapper(model, grid_size, d_state, device, cfg=config)
        
    elif engine_type == 'QUANTUM':
        if PENNYLANE_AVAILABLE:
            if not isinstance(model, QuantumKernel):
                logging.info("Creando QuantumKernel para motor híbrido...")
                model = QuantumKernel(n_qubits=9, n_layers=2)
                model = model.to(device)
        return HybridMotorWrapper(model, grid_size, d_state, device, cfg=config)
        
    else: # CARTESIAN / STANDARD
        return Aetheria_Motor(model, grid_size, d_state, device, cfg=config)
