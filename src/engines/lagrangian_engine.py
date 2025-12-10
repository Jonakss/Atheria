
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from src.engines.base_engine import EngineProtocol
# from src.models.lagrangian_net import LagrangianNetwork # Imported dynamically or assumed available

class LagrangianEngine:
    def __init__(self, config: Dict[str, Any], model: Optional[nn.Module] = None):
        self.cfg = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.dt = config.get('dt', 0.01)
        self.grid_size = config.get('grid_size', 64)
        self.d_state = config.get('d_state', 3) # Channels
        
        # Estado: q (pos) y v (vel)
        # Inicializamos aleatoriamente
        shape = (1, self.d_state, self.grid_size, self.grid_size)
        self.q = torch.randn(shape, device=self.device) * 0.1
        self.v = torch.randn(shape, device=self.device) * 0.1
        
        # Modelo
        if model:
            self.model = model
        else:
            # Fallback si no se pasa modelo externo, útil para tests
            from src.models.lagrangian_net import LagrangianNetwork
            self.model = LagrangianNetwork(state_dim=self.d_state).to(self.device)
            
    # ---------------------
    # IMPLEMENTACIÓN PROTOCOLO
    # ---------------------
    
    @property
    def state(self):
        # El protocolo espera un objeto con .psi
        # Devolvemos un objeto dummy o wrapper
        class StateWrapper:
            def __init__(self, q):
                # Convertimos q real a complejo para visualizar como magnitud/fase
                # O simplemente mapeamos q -> real, v -> imag
                # Shape [B, H, W, C] (Channels Last para visualización)
                self.psi = q.permute(0, 2, 3, 1).contiguous().to(torch.complex64)
        return StateWrapper(self.q)

    @state.setter
    def state(self, new_state):
        # new_state tiene .psi [B, H, W, C]
        # Convertimos back a [B, C, H, W]
        # Asumimos que solo seteamos 'q', y 'v' se resetea o mantiene
        psi = new_state.psi
        if psi.shape[-1] != self.d_state:
             # Ajustar canales si es necesario
             pass
        q_new = psi.real.permute(0, 3, 1, 2).to(self.device)
        self.q = q_new
        self.v = torch.zeros_like(self.q) # Reset velocidad al forzar estado

    def compile_model(self):
        self.model = torch.compile(self.model)

    def get_model_for_params(self):
        return self.model

    def get_initial_state(self, batch_size=1):
        return torch.randn(batch_size, self.d_state, self.grid_size, self.grid_size, device=self.device)

    def get_dense_state(self, roi=None, check_pause_callback=None):
        # Retorna tensor Channels Last [B, H, W, C] complex
        # Mapeo: q -> Real, v -> Imag (para ver "fase" como velocidad)
        
        complex_state = torch.complex(self.q, self.v)
        return complex_state.permute(0, 2, 3, 1).detach() # [B, H, W, C]

    def get_visualization_data(self, viz_type="density"):
        # Protocolo standard
        # Normalizamos [0, 1]
        data = self.get_dense_state() # [B, H, W, C] complex
        
        if viz_type == "density":
            field = data.abs().pow(2).sum(dim=-1).sqrt() # Magnitud
        elif viz_type == "phase":
            field = data.angle().mean(dim=-1) # Fase promedio
        elif viz_type == "real":
            field = data.real.mean(dim=-1)
        else:
            field = data.abs().mean(dim=-1)
            
        # Normalize simple
        f_min, f_max = field.min(), field.max()
        if f_max > f_min:
            field = (field - f_min) / (f_max - f_min)
        else:
            field = torch.zeros_like(field)
            
        return {
            "data": field.cpu().numpy(),
            "type": viz_type, 
            "shape": list(field.shape),
            "min": 0.0,
            "max": 1.0,
            "engine": "LagrangianEngine"
        }

    def apply_tool(self, action, params):
        # Implementar herramientas básicas si es necesario
        pass

    # ---------------------
    # CORE PHYSICS (EULER-LAGRANGE)
    # ---------------------

    def evolve_internal_state(self, step=None):
        if not hasattr(self, 'integrator'):
            from src.physics.variational_integrator import VariationalIntegrator
            self.integrator = VariationalIntegrator(self.model)
            
        self.q, self.v = self.integrator.step(self.q, self.v, self.dt)

    def evolve_step(self, current_psi):
        if not hasattr(self, 'integrator'):
            from src.physics.variational_integrator import VariationalIntegrator
            self.integrator = VariationalIntegrator(self.model)

        q = current_psi.real.permute(0, 3, 1, 2)
        v = current_psi.imag.permute(0, 3, 1, 2)
        
        next_q, next_v = self.integrator.step(q, v, self.dt)
        
        next_psi = torch.complex(next_q, next_v).permute(0, 2, 3, 1)
        return next_psi

    # Manual solve_euler_lagrange removed in favor of VariationalIntegrator


