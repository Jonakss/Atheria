import torch
import torch.nn as nn
import numpy as np
import logging

from .holographic_mixin import HolographicMixin
from qca.observer_effect import ObserverKernel

class LatticeEngine(HolographicMixin):
    """
    Motor de Simulación para Lattice Gauge Theory (Phase 4) y Volumetric ORT (Phase 5).
    Simula campos de gauge SU(3) en 2D+1 y Campos ORT 37D Volumétricos.
    """
    def __init__(self, grid_size, d_state=37, depth=32, device='cpu', group='SU3', beta=6.0):
        self.grid_size = grid_size
        self.depth = depth
        self.d_state = d_state # Phase 5: 37 Dimensions
        self.device = device
        self.group = group
        self.beta = beta
        
        # SU(3) matrices are 3x3 complex
        self.N = 3
        
        # Pre-compute identity for efficiency
        self.identity = torch.eye(self.N, dtype=torch.complex64, device=self.device)
        
        # Initialize links: [2, H, W, 3, 3] (Complex)
        # 2 directions (Right, Down), HxW grid, 3x3 SU(3) matrices
        self.config = {
            'grid_size': grid_size,
            'depth': depth,
            'd_state': d_state,
            'beta': beta
        }
        self.t = 0
        self.dt = 0.01

        # holographic depth default (Phase 4 legacy)
        self.bulk_depth = 8
        
        # Phase 5: Observer Kernel
        self.observer = ObserverKernel(high_res_dim=(d_state, depth, grid_size, grid_size))
        
        self.reset()
        
        # Gateway Process: Click-Out Mechanism
        self.click_out_enabled = False
        self.click_out_chance = 0.01 # Probability of tunneling event per step

        logging.info(f"🌌 LatticeEngine (SU3 + ORT) inicializado: Grid={grid_size}x{grid_size}x{depth}, Ch={d_state}")

    def reset(self):
        """Reinicia el estado del retículo."""
        self.step_count = 0
        self.t = 0
        self.links = self._initialize_links()
        
        # Phase 5: Initialize 37D ORT State (The "Fog")
        # Shape: [B=1, C=37, D, H, W]
        self.ort_state = torch.zeros(1, self.d_state, self.depth, self.grid_size, self.grid_size, device=self.device)
        # Add some initial noise
        self.ort_state = torch.randn_like(self.ort_state) * 0.1
        
        logging.info("🌌 LatticeEngine reseteado.")

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
        # dims of Ux: 0=B, 1=H, 2=W, 3=N, 4=N
        
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
        trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
        action_density = - (self.beta / self.N) * trace.real
        return action_density.sum()

    def step(self):
        """
        Main simulation step.
        Phase 4: Metropolis-Hastings for SU(3).
        Phase 5: Observer Effect + ORT Dynamics.
        """
        self.step_count += 1
        
        # --- Phase 4: SU(3) Updates ---
        # 1. Propose update
        R = self._generate_random_su3(epsilon=0.2) # Step size
        proposed_links = self._matmul(R, self.links)
        
        # 2. Compute Action Change (Approximate Global Update)
        U_p_curr = self._compute_plaquettes(self.links)
        S_local_curr = - (self.beta / self.N) * torch.diagonal(U_p_curr, dim1=-2, dim2=-1).sum(-1).real
        
        U_p_prop = self._compute_plaquettes(proposed_links)
        S_local_prop = - (self.beta / self.N) * torch.diagonal(U_p_prop, dim1=-2, dim2=-1).sum(-1).real
        
        delta_S_local = S_local_prop - S_local_curr
        
        # Metropolis probability
        prob = torch.exp(-delta_S_local)
        rand = torch.rand_like(prob)
        accept_mask = (rand < prob).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(self.links)
        
        self.links = torch.where(accept_mask, proposed_links, self.links)

        # --- Phase 5: ORT & Observer Effect ---
        # 1. Get Observer Mask (Cone of Vision)
        obs_mask = self.observer.get_observer_mask(self.ort_state.shape, viewport_center=None) # Default center
        
        # 2. Collapse Fog -> Reality
        # active_state = self.observer.collapse(self.ort_state, obs_mask)
        
        # 3. Apply Dynamics (Placeholder for UNet3D / Hamiltonian)
        # For now, we just diffuse the state slightly to simulate "life"
        # Diffusion in 3D: laplacian
        # Minimal 3D diffusion kernel
        # self.ort_state += 0.01 * torch.randn_like(self.ort_state) # Brownian motion
        # Apply mask to keep unobserved regions "foggy" (or static/statistical)
        
        # Simulating "Activity" in the observed region:
        noise = torch.randn_like(self.ort_state) * 0.1
        self.ort_state = self.ort_state + (noise * obs_mask)
        
        # Clamp to reasonable range
        self.ort_state = torch.clamp(self.ort_state, -1.0, 1.0)
        
        
        # 4. Gateway Process
        if self.click_out_enabled:
            self._apply_click_out()
        
        return self.links

    def get_visualization_data(self, viz_type="density"):
        """
        Retorna datos para visualización.
        Phase 5 Update: Supports 'volumetric' 3D data.
        """
        try:
            if viz_type == "density":
                # Energy Density = 1 - 1/N Re Tr P (2D Projection)
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                energy = 1.0 - (1.0 / self.N) * trace.real
                data = energy.squeeze(0)  # [H, W]
                shape = list(data.shape)
                channels = 1
                
            elif viz_type == "holographic":
                # Holographic RGB from SU(3)
                U_p = self._compute_plaquettes(self.links)
                diag = torch.diagonal(U_p, dim1=-2, dim2=-1)
                data = diag.real.squeeze(0) # [H, W, 3]
                data = 1.0 - data # Contrast
                shape = list(data.shape)
                channels = 3
                
            elif viz_type == "volumetric" or viz_type == "holographic_volumetric":
                # Phase 5: Return full 3D state (or a channel of it)
                # self.ort_state is [1, 37, D, H, W]
                # Frontend likely wants [D, H, W] (scalar density) or [D, H, W, 3] (RGB)
                
                # Let's map 3 principal channels (Mag, Phase, Topo) to RGB?
                # Or just return density of first channel for MVP.
                
                # Taking channel 0 as "Matter Density"
                # Shape: [D, H, W]
                data = self.ort_state[0, 0, :, :, :]
                
                # Normalize 0-1
                data = (data - data.min()) / (data.max() - data.min() + 1e-6)
                shape = list(data.shape)
                channels = 1
                
            else:
                # Default fallback
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                energy = 1.0 - (1.0 / self.N) * trace.real
                data = energy.squeeze(0)
                shape = list(data.shape)
                channels = 1
            
            # Convertir a numpy
            data_np = data.cpu().numpy().astype(np.float32)
            
            # Additional Norm if needed (handled inside blocks mostly)
            if viz_type not in ["volumetric", "holographic_volumetric"] and viz_type != "holographic":
                 min_val, max_val = float(data_np.min()), float(data_np.max())
                 if max_val > min_val:
                     data_np = (data_np - min_val) / (max_val - min_val)
            
            return {
                "data": data_np,
                "type": viz_type,
                "shape": shape,
                "min": 0.0,
                "max": 1.0,
                "engine": "LatticeEngine",
                "channels": channels
            }
            
        except Exception as e:
            logging.error(f"Error en get_visualization_data: {e}")
            return {"data": None, "type": viz_type, "error": str(e)}

    def compile_model(self):
        pass

    def get_model_for_params(self):
        return None

    def get_initial_state(self, batch_size=1):
        # LatticeEngine maneja su propio estado interno
        return torch.zeros(batch_size, 1, self.grid_size, self.grid_size, device=self.device)

    def evolve_step(self, current_psi):
        self.step()
        return current_psi

    def evolve_internal_state(self, step=None):
        self.step()

    def get_dense_state(self, roi=None, check_pause_callback=None):
        viz_dict = self.get_visualization_data("density")
        energy_np = viz_dict['data']
        energy = torch.tensor(energy_np, device=self.device, dtype=torch.float32)
        energy = energy.unsqueeze(0).unsqueeze(-1)
        return torch.complex(energy, torch.zeros_like(energy))

    def apply_tool(self, action, params):
        logging.info(f"🛠️ LatticeEngine aplicando herramienta: {action} | Params: {params}")
        if action == 'collapse':
            # Phase 5: Trigger manual collapse/observation at a point?
            intensity = float(params.get('intensity', 0.5))
            center = None
            if 'x' in params and 'y' in params:
                center = (int(params['y']), int(params['x']))
            self._apply_collapse(intensity, center)
            return True
        elif action == 'vortex':
            x = int(params.get('x', self.grid_size // 2))
            y = int(params.get('y', self.grid_size // 2))
            radius = int(params.get('radius', 5))
            strength = float(params.get('strength', 1.0))
            self._apply_vortex(x, y, radius, strength)
            return True
        elif action == 'wave':
            k_x = float(params.get('k_x', 1.0))
            k_y = float(params.get('k_y', 1.0))
            self._apply_wave(k_x, k_y)
            return True
        elif action == 'set_click_out':
            self.click_out_enabled = bool(params.get('enabled', False))
            self.click_out_chance = float(params.get('chance', 0.01))
            logging.info(f"🌀 Click-Out Config: Enabled={self.click_out_enabled}, Chance={self.click_out_chance}")
            return True
        else:
            logging.warning(f"⚠️ Herramienta no soportada por LatticeEngine: {action}")
            return False

    def _apply_collapse(self, intensity, center=None):
        # Applies noise to SU(3) links (Phase 4 Logic)
        noise = self._generate_random_su3(epsilon=intensity * 2.0)
        if center:
            cy, cx = center
            radius = int(self.grid_size * 0.15) 
            y, x = torch.meshgrid(torch.arange(self.grid_size, device=self.device), torch.arange(self.grid_size, device=self.device), indexing='ij')
            dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
            mask = (dist < radius).float().view(1, 1, self.grid_size, self.grid_size, 1, 1)
            perturbed_links = self._matmul(noise, self.links)
            self.links = (1 - mask) * self.links + mask * perturbed_links
            
            # Phase 5: Also perturb 3D state
            # Simple circular cylinder poke
            self.ort_state[:, :, :, cy-5:cy+5, cx-5:cx+5] += intensity
        else:
            self.links = self._matmul(noise, self.links)
            self.ort_state += intensity

    def _apply_vortex(self, x, y, radius, strength):
        yy, xx = torch.meshgrid(torch.arange(self.grid_size, device=self.device), torch.arange(self.grid_size, device=self.device), indexing='ij')
        xx = xx - x
        yy = yy - y
        theta = torch.atan2(yy, xx)
        phase = strength * theta
        
        u00 = torch.exp(1j * phase)
        u11 = torch.exp(-1j * phase)
        u22 = torch.ones_like(phase, dtype=torch.complex64)
        
        G = torch.zeros(self.grid_size, self.grid_size, self.N, self.N, dtype=torch.complex64, device=self.device)
        G[..., 0, 0] = u00
        G[..., 1, 1] = u11
        G[..., 2, 2] = u22
        
        G = G.view(1, 1, self.grid_size, self.grid_size, self.N, self.N)
        G_px = torch.roll(G, shifts=-1, dims=3)
        G_py = torch.roll(G, shifts=-1, dims=2)
        
        Ux = self.links[:, 0:1]
        Uy = self.links[:, 1:2]
        Ux_new = torch.matmul(torch.matmul(G, Ux), G_px.conj().transpose(-2, -1))
        Uy_new = torch.matmul(torch.matmul(G, Uy), G_py.conj().transpose(-2, -1))
        
        dist = torch.sqrt(xx**2 + yy**2)
        mask = torch.sigmoid((radius - dist) * 0.5).view(1, 1, self.grid_size, self.grid_size, 1, 1)
        
        self.links[:, 0:1] = (1 - mask) * Ux + mask * Ux_new
        self.links[:, 1:2] = (1 - mask) * Uy + mask * Uy_new

    def _apply_click_out(self):
        if np.random.random() > self.click_out_chance:
            return
        n_tunnels = int(self.grid_size * 0.5)
        b = 0
        src_y = torch.randint(0, self.grid_size, (n_tunnels,), device=self.device)
        src_x = torch.randint(0, self.grid_size, (n_tunnels,), device=self.device)
        dst_y = torch.randint(0, self.grid_size, (n_tunnels,), device=self.device)
        dst_x = torch.randint(0, self.grid_size, (n_tunnels,), device=self.device)
        
        src_links = self.links[b, :, src_y, src_x].clone()
        dst_links = self.links[b, :, dst_y, dst_x].clone()
        
        self.links[b, :, src_y, src_x] = dst_links
        self.links[b, :, dst_y, dst_x] = src_links

    def _apply_wave(self, k_x, k_y):
        y, x = torch.meshgrid(torch.arange(self.grid_size, device=self.device), torch.arange(self.grid_size, device=self.device), indexing='ij')
        phase = (k_x * x + k_y * y).float()
        epsilon = 0.3
        phi = epsilon * torch.sin(phase / 10.0)
        u_wave = torch.zeros(self.grid_size, self.grid_size, self.N, self.N, dtype=torch.complex64, device=self.device)
        u_wave[..., 0, 0] = torch.exp(1j * phi)
        u_wave[..., 1, 1] = torch.exp(-1j * phi)
        u_wave[..., 2, 2] = 1.0
        u_wave = u_wave.view(1, 1, self.grid_size, self.grid_size, self.N, self.N)
        self.links = torch.matmul(u_wave, self.links)


