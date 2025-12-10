
import torch
import torch.nn as nn
from typing import Tuple, Callable

class VariationalIntegrator:
    """
    Integrador Variacional Reutilizable.
    
    Implementa la resolución de las ecuaciones de Euler-Lagrange para cualquier
    modelo diferenciable que retorne un escalar (Lagrangiano).
    
    Uso:
        integrator = VariationalIntegrator(model_fn)
        q_next, v_next = integrator.step(q, v, dt)
    """
    def __init__(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Args:
            model: Función o red que toma (q, v) y retorna Scalar L.
                   L = model(q, v).sum()
        """
        self.model = model

    def step(self, q: torch.Tensor, v: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza un paso de integración usando Euler-Lagrange semi-implícito.
        
        Args:
            q: General coordinates [B, C, H, W] or similar.
            v: General velocities (same shape as q).
            dt: Time step.
            
        Returns:
            (q_next, v_next)
        """
        # Habilitar gradientes para q y v temporales
        q = q.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        
        # 1. Forward Pass: Calcular L
        # Asumimos que model retorna densidad, sumamos para tener escalar total "Action Rate"
        L_density = self.model(q, v) 
        L_total = L_density.sum()
        
        # 2. Gradientes Primeros (dL/dq, dL/dv)
        grads = torch.autograd.grad(L_total, [q, v], create_graph=True)
        dL_dq = grads[0]
        dL_dv = grads[1]
        
        # 3. Calcular Aceleración via inversa del Hessiano (Aproximación Diagonal)
        # a = (d2L/dv2)^-1 * (dL/dq - d2L/dvdq * v)
        
        # Hessian Diagonal Approximation for d2L/dv2
        # hessian_diag_approx = d(dL_dv)/dv * 1
        hessian_diag_approx = torch.autograd.grad(dL_dv, v, grad_outputs=torch.ones_like(dL_dv), retain_graph=True)[0]
        
        # Mass Inverse (adding epsilon for numerical stability)
        mass_inv = 1.0 / (hessian_diag_approx + 1e-6)
        
        # Mixed Term Approximation: d2L/dvdq * v
        # Calculates directional derivative of dL/dv along v with respect to q
        mixed_term = torch.autograd.grad(dL_dv, q, grad_outputs=v, retain_graph=True)[0]
        
        # Solve for acceleration
        # Euler-Lagrange Equation: d/dt (dL/dv) = dL/dq
        # d/dt (dL/dv) = (d2L/dv2) * a + (d2L/dvdq) * v = dL/dq
        # => a = (d2L/dv2)^-1 * (dL/dq - (d2L/dvdq) * v)
        accel = mass_inv * (dL_dq - mixed_term)
        
        # Integración Semi-Implicita (Symplectic flow)
        # v_{t+1} = v_t + a * dt
        # q_{t+1} = q_t + v_{t+1} * dt  <-- Updating q with NEW velocity improves energy conservation
        
        v_next = v + accel * dt
        q_next = q + v_next * dt
        
        return q_next.detach(), v_next.detach()
