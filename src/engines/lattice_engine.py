import torch
import torch.nn as nn
import numpy as np
import logging

class LatticeEngine:
    """
    Motor de Simulación para Lattice Gauge Theory (Phase 4).
    Simula campos de gauge SU(3) en un retículo espacio-temporal 2D+1.
    """
    def __init__(self, grid_size, d_state, device='cpu', group='SU3', beta=6.0):
        self.grid_size = grid_size
        self.device = device
        self.group = group
        self.beta = beta
        
        # SU(3) matrices are 3x3 complex
        self.N = 3
        
        # Pre-compute identity for efficiency
        self.identity = torch.eye(self.N, dtype=torch.complex64, device=self.device).view(1, 1, 1, 1, self.N, self.N)
        
        self.step_count = 0
        self.links = self._initialize_links()
        
        logging.info(f"🌌 LatticeEngine (SU3) inicializado: Grid={grid_size}x{grid_size}, Beta={beta}")

    def _initialize_links(self):
        """Inicializa los links del retículo (Cold Start)."""
        # Cold Start: Todos los links son Identidad
        # Shape: [1, 2, H, W, 3, 3]
        links = torch.eye(self.N, dtype=torch.complex64, device=self.device)
        links = links.view(1, 1, 1, 1, self.N, self.N).repeat(1, 2, self.grid_size, self.grid_size, 1, 1)
        
        # Add small noise to break symmetry immediately
        noise = self._generate_random_su3(epsilon=0.1)
        return self._matmul(noise, links)

    def _generate_random_su3(self, epsilon=0.1):
        """Genera matrices SU(3) aleatorias cerca de la identidad."""
        # Generadores de SU(3) (Matrices de Gell-Mann simplificadas para updates)
        # Algebra su(3): matrices anti-hermitianas sin traza
        
        # Random hermitian matrix H
        H = torch.randn(1, 2, self.grid_size, self.grid_size, self.N, self.N, dtype=torch.complex64, device=self.device)
        H = H + H.conj().transpose(-2, -1)
        
        # Make traceless
        tr = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1, keepdim=True)
        H = H - (tr / self.N).unsqueeze(-1) * self.identity
        
        # U = exp(i * epsilon * H)
        # Para epsilon pequeño, aproximamos: U = I + i * epsilon * H
        # Mejor usar matrix_exp para garantizar unitariedad
        X = 1j * epsilon * H
        return torch.linalg.matrix_exp(X)

    def _matmul(self, A, B):
        """Multiplicación de matrices batch."""
        return torch.matmul(A, B)

    def _compute_plaquettes(self, links):
        """
        Calcula la plaqueta U_p para cada sitio.
        U_p = U_x(n) * U_y(n+x) * U_x^dag(n+y) * U_y^dag(n)
        """
        # Desempaquetar links
        Ux = links[:, 0] # [B, H, W, N, N]
        Uy = links[:, 1]
        
        # Shift links to get neighbors
        # U_y(n+x): Roll Ux en eje W (x) -1
        Uy_px = torch.roll(Uy, shifts=-1, dims=2) # dims=2 es W (después de Batch) -> No, dims son [B, H, W, N, N] -> H=1, W=2
        # Espera, shape es [B, 2, H, W, N, N] -> Ux es [B, H, W, N, N]
        # Dims de Ux: 0=B, 1=H, 2=W, 3=N, 4=N
        
        Uy_px = torch.roll(Uy, shifts=-1, dims=2) # Shift en W
        Ux_py = torch.roll(Ux, shifts=-1, dims=1) # Shift en H
        
        # Dagger (Conjugate Transpose)
        Ux_py_dag = Ux_py.conj().transpose(-2, -1)
        Uy_dag = Uy.conj().transpose(-2, -1)
        
        # Plaquette calculation: Ux * Uy_px * Ux_py_dag * Uy_dag
        # Order matters!
        U_top = torch.matmul(Ux, Uy_px)
        U_bottom = torch.matmul(Ux_py_dag, Uy_dag)
        U_p = torch.matmul(U_top, U_bottom)
        
        return U_p

    def _compute_action(self, links):
        """
        Calcula la Acción de Wilson S.
        S = -beta/N * Sum Re(Tr(U_p))
        """
        U_p = self._compute_plaquettes(links)
        
        # Trace: Sum of diagonal elements
        trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
        
        # Real part
        action_density = - (self.beta / self.N) * trace.real
        
        return action_density.sum()

    def step(self):
        """
        Metropolis-Hastings update step.
        """
        self.step_count += 1
        
        # 1. Propose update
        # U' = R * U
        R = self._generate_random_su3(epsilon=0.2) # Step size
        proposed_links = self._matmul(R, self.links)
        
        # 2. Compute Action Change
        # Delta S = S(U') - S(U)
        # Optimization: Local updates are better, but global batch update is easier in PyTorch
        # We will use a "Checkerboard" or global update with acceptance mask
        
        current_action = self._compute_action(self.links)
        proposed_action = self._compute_action(proposed_links)
        
        delta_S = proposed_action - current_action
        
        # 3. Accept/Reject (Global approximation for prototype)
        # Correct Metropolis requires local delta_S calculation per site.
        # For this prototype, we'll implement a simplified "Global Hybrid Monte Carlo" style step
        # or just accept if action decreases (Gradient Flow-ish) to ensure stability first.
        
        # Let's do a proper local Metropolis check vectorised
        # We need local action density, not sum
        
        U_p_curr = self._compute_plaquettes(self.links)
        S_local_curr = - (self.beta / self.N) * torch.diagonal(U_p_curr, dim1=-2, dim2=-1).sum(-1).real
        
        U_p_prop = self._compute_plaquettes(proposed_links)
        S_local_prop = - (self.beta / self.N) * torch.diagonal(U_p_prop, dim1=-2, dim2=-1).sum(-1).real
        
        # Delta S per site (approximate, since changing a link affects neighbors)
        # This is a "Global Update" proposal, which has low acceptance rate.
        # But for visualization purposes of "flow", it works.
        
        delta_S_local = S_local_prop - S_local_curr
        
        # Metropolis probability
        prob = torch.exp(-delta_S_local)
        rand = torch.rand_like(prob)
        
        # Accept mask [B, H, W]
        accept = rand < prob
        
        # Expand mask to [B, 2, H, W, N, N]
        accept_mask = accept.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(self.links)
        
        # Update links
        self.links = torch.where(accept_mask, proposed_links, self.links)
        
        return self.links

    def get_visualization_data(self, viz_type="density"):
        """
        Retorna datos para visualización.
        """
        if viz_type == "density":
            # Energy Density = 1 - 1/N Re Tr P
            U_p = self._compute_plaquettes(self.links)
            trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
            energy = 1.0 - (1.0 / self.N) * trace.real
            return energy.squeeze(0) # [H, W]
            
        return None

    def compile_model(self):
        """
        Compila el modelo para optimización (no-op para LatticeEngine).
        """
        pass

    def get_model_for_params(self):
        """
        Retorna el modelo para contar parámetros (None para LatticeEngine).
        """
        return None

    def get_initial_state(self, batch_size=1):
        """
        Retorna el estado inicial (dummy para LatticeEngine).
        """
        # LatticeEngine maneja su propio estado interno (self.links)
        # Retornamos un tensor dummy para satisfacer la API del trainer
        return torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=self.device)

    def evolve_step(self, current_psi):
        """
        Evoluciona el estado un paso (Metropolis-Hastings para Lattice).
        """
        # LatticeEngine evoluciona self.links internamente
        self.step()
        # Retornamos el mismo dummy state
        return current_psi

    def evolve_internal_state(self, step=None):
        """
        Alias para step() para compatibilidad.
        """
        self.step()

