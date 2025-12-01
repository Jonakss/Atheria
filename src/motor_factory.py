import torch
import logging
import torch.nn as nn
from types import SimpleNamespace

from .engines.qca_engine import Aetheria_Motor, QuantumState
from .qca_engine_polar import PolarEngine, QuantumStatePolar
# Try importing PennyLane engine, handle if not installed
try:
    from .qca_engine_pennylane import QuantumKernel
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

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
            # Pero idealmente debería ser ya un PolarEngine o compatible
            pass

    def _evolve_logic(self, psi_in):
        """
        Sobrescribe la lógica de evolución para usar coordenadas polares.
        """
        # 1. Convertir estado cartesiano (psi_in) a Polar
        # psi_in es (1, H, W, d_state) complejo
        # PolarEngine espera (B, 1, H, W) para magnitud y fase (asumiendo d_state=1 por ahora o adaptando)
        
        # NOTA: PolarEngine en qca_engine_polar.py parece diseñado para 1 canal (magnitud, fase).
        # Aetheria usa d_state canales.
        # Por ahora, adaptamos para el primer canal o hacemos broadcast.
        
        # Extraer magnitud y fase del estado complejo
        magnitude = psi_in.abs().permute(0, 3, 1, 2) # (1, d_state, H, W)
        phase = psi_in.angle().permute(0, 3, 1, 2)   # (1, d_state, H, W)
        
        # PolarEngine espera (B, 1, H, W). Si d_state > 1, esto podría ser un problema si PolarEngine no lo soporta.
        # Asumiremos que PolarEngine puede manejar d_state canales o iteramos.
        # Viendo el código de PolarEngine, usa conv2d que soporta canales.
        
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
        self.last_delta_psi = new_psi - psi_in # Delta aproximado
        
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
        # QuantumKernel espera (B, C, H, W)
        x = psi_in.abs().permute(0, 3, 1, 2) # Usamos magnitud como input "clásico" al circuito
        
        # Forward pass
        # output es (B, n_actions, H, W). Asumimos n_actions mapea a cambios en el estado.
        out = self.operator(x)
        
        # Interpretar output como actualización de estado
        # Esto es muy experimental. Por ahora, simplemente sumamos al estado actual como una "fuerza"
        # Asumimos que out tiene mismos canales que x
        
        delta = out.permute(0, 2, 3, 1)
        
        # Aplicar actualización (simple Euler)
        # Convertir a complejo (fase 0 por simplicidad o aleatoria)
        delta_complex = torch.complex(delta, torch.zeros_like(delta))
        
        new_psi = psi_in + delta_complex * 0.1 # Factor de aprendizaje/tiempo
        
        # Normalizar
        norm = torch.sqrt(torch.sum(new_psi.abs().pow(2), dim=-1, keepdim=True))
        new_psi = new_psi / (norm + 1e-9)
        
        self.last_psi_input = psi_in
        self.last_delta_psi = delta_complex
        
        return new_psi

def get_motor(config, device, model=None):
    """
    Factory method para obtener la instancia del motor adecuada.
    
    Args:
        config: Objeto de configuración (SimpleNamespace o dict)
        device: Dispositivo torch
        model: Modelo (nn.Module) opcional ya instanciado.
               Si es None, se intentará instanciar según config.
    
    Returns:
        Instancia de Aetheria_Motor (o subclase)
    """
    # Determinar tipo de motor
    engine_type = 'CARTESIAN'
    if hasattr(config, 'ENGINE_TYPE'):
        engine_type = config.ENGINE_TYPE
    elif isinstance(config, dict) and 'ENGINE_TYPE' in config:
        engine_type = config['ENGINE_TYPE']
        
    logging.info(f"🏭 Motor Factory: Solicitado motor tipo '{engine_type}'")
    
    # Parámetros comunes
    grid_size = getattr(config, 'GRID_SIZE_TRAINING', 64)
    if hasattr(config, 'GRID_SIZE_INFERENCE'): # Prioridad a inferencia si existe en contexto de inferencia
         # Detectar contexto (si estamos en inferencia, config suele tener GRID_SIZE_INFERENCE)
         # Pero cuidado, config puede ser exp_cfg de entrenamiento.
         # Usaremos el valor pasado explícitamente si existe, sino defaults.
         pass
         
    # Extraer d_state
    d_state = 2
    model_params = getattr(config, 'MODEL_PARAMS', None)
    if model_params:
        if isinstance(model_params, dict):
            d_state = model_params.get('d_state', 2)
        else:
            d_state = getattr(model_params, 'd_state', 2)
            
    # Instanciar el modelo si no viene dado
    if model is None:
        from .model_loader import instantiate_model
        # Asegurar que config tiene lo necesario para instantiate_model
        try:
            model = instantiate_model(config)
        except Exception as e:
            logging.error(f"Error instanciando modelo base: {e}")
            raise

    # Seleccionar Wrapper
    if engine_type == 'POLAR':
        # Para Polar, necesitamos un PolarEngine. 
        # Si el modelo cargado NO es PolarEngine, lo envolvemos en uno
        if not isinstance(model, PolarEngine):
            logging.info("Adaptando modelo estándar a PolarEngine...")
            model = PolarEngine(model, grid_size=grid_size)
            
        return PolarMotorWrapper(model, grid_size, d_state, device, cfg=config)
        
    elif engine_type == 'QUANTUM':
        # Para Quantum, necesitamos QuantumKernel
        if PENNYLANE_AVAILABLE:
            if not isinstance(model, QuantumKernel):
                logging.info("Creando QuantumKernel para motor híbrido...")
                # Aquí hay un gap: QuantumKernel es un modelo distinto, no envuelve a una UNet fácilmente.
                # Para este POC, instanciamos un QuantumKernel nuevo ignorando el modelo pasado si no es compatible,
                # o usamos el modelo pasado como parte clásica.
                # Por simplicidad para el POC:
                model = QuantumKernel(n_qubits=9, n_layers=2) # 3x3 kernel
                model = model.to(device)
                
        return HybridMotorWrapper(model, grid_size, d_state, device, cfg=config)
        
    else: # CARTESIAN / STANDARD
        return Aetheria_Motor(model, grid_size, d_state, device, cfg=config)
