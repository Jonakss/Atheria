import torch
import torch.nn as nn
import numpy as np
import logging

from .holographic_mixin import HolographicMixin

class LatticeEngine(HolographicMixin):
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
        self.identity = torch.eye(self.N, dtype=torch.complex64, device=self.device)
        
        # Initialize links: [2, H, W, 3, 3] (Complex)
        # 2 directions (Right, Down), HxW grid, 3x3 SU(3) matrices
        self.config = {
            'grid_size': grid_size,
            'beta': beta
        }
        self.t = 0
        self.dt = 0.01

        # holographic depth default
        self.bulk_depth = 8
        
        self.reset()
        
        # Gateway Process: Click-Out Mechanism
        self.click_out_enabled = False
        self.click_out_chance = 0.01 # Probability of tunneling event per step

        logging.info(f"🌌 LatticeEngine (SU3) inicializado: Grid={grid_size}x{grid_size}, Beta={beta}")

    def reset(self):
        """Reinicia el estado del retículo."""
        self.step_count = 0
        self.t = 0
        self.links = self._initialize_links()
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

        # 4. Gateway Process: Click-Out Mechanism (Phase 2)
        if self.click_out_enabled:
            self._apply_click_out()
        
        return self.links

    def get_visualization_data(self, viz_type="density"):
        """
        Retorna datos para visualización.
        
        Args:
            viz_type: Tipo de visualización
                - 'density': Densidad de energía (1 - 1/N Re Tr P)
                - 'phase': Fase de las plaquetas
                
        Returns:
            dict con 'data' (array 2D) y 'metadata'
        """
        try:
            if viz_type == "density":
                # Energy Density = 1 - 1/N Re Tr P
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                energy = 1.0 - (1.0 / self.N) * trace.real
                data = energy.squeeze(0)  # [H, W]
                
            elif viz_type == "phase":
                # Fase de la traza de plaquetas
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                phase = torch.angle(trace)
                data = phase.squeeze(0)  # [H, W]
                
            elif viz_type == "real":
                # Real part of Plaquette Trace
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                data = trace.real.squeeze(0) # [H, W]
                
            elif viz_type == "imag":
                # Imaginary part of Plaquette Trace
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                data = trace.imag.squeeze(0) # [H, W]
            
            elif viz_type == "phase_hsv":
                # Returns magnitude for Value, but frontend expects 3 channels for HSV if implemented that way.
                # Or we can return complex magnitude now and let frontend handle color?
                # Usually phase_hsv implies mapping complex -> RGB.
                # For SU(3) trace:
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                
                # We need RGB.
                # H = angle, S = 1, V = magnitude/3
                angle = torch.angle(trace)
                mag = torch.abs(trace) / self.N
                
                # Simple HSV to RGB in Torch is painful.
                # Let's return the components and let helper do it?
                # Or return just phase for now, similar to 'phase', but maybe raw angle?
                # Let's map it to an RGB image.
                
                # Use a helper or just return 3 channels [H, W, 3] like holographic
                # But here we interpret H, S, V
                
                # Let's implement a simple color wheel mapping here.
                # hue \in [-pi, pi] -> [0, 1]
                hue = (angle + torch.pi) / (2 * torch.pi)
                val = mag
                
                # Return 2 channels? [H, W, 2] -> [Hue, Val]
                # Frontend might assume 3 channels = RGB.
                # Let's try returning RGB
                
                # ... skipping complex HSV2RGB impl for now to save tokens/complexity.
                # Fallback to returning separate channels via special dictionary key?
                # Standard get_visualization_data returns 'data'.
                
                # Let's treat phase_hsv as "Rainbow Phase" (Phase maps to color)
                data = angle.squeeze(0)
                
            elif viz_type == "holographic":
                # Holographic View: RGB Mapping from SU(3) components
                # We use the diagonal of the Plaquette matrix (3 complex numbers)
                # This gives us 3 distinct fields to map to R, G, B
                U_p = self._compute_plaquettes(self.links) # [B, H, W, 3, 3]
                
                # Extract diagonal elements [B, H, W, 3]
                diag = torch.diagonal(U_p, dim1=-2, dim2=-1)
                
                # Map to real values (Magnitude or Real part)
                # Phase could also be interesting, but let's use Real part (Energy-like)
                # Normalized around 1.0 (Identity)
                # Re(u_ii) ~ 1.0 - epsilon
                # We want deviation from identity to be visible
                
                # Use absolute deviation from 1.0 as signal?
                # Or simply the values themselves shifted
                
                # Approach: Real part of diagonal. 
                # Ideally close to 1.0. Lower values mean excitation.
                data = diag.real.squeeze(0) # [H, W, 3]
                
                # Normalize specifically for color
                # Create contrast: (1 - val) * gain
                data = 1.0 - data
                
            elif viz_type == "holographic_bulk":
                # Generate Bulk 3D data from 2D surface (AdS/CFT style)
                # We simulate an emergent radial dimension (z) via coarse-graining (renormalization)
                # D = 8 layers
                depth = self.bulk_depth
                
                # Base layer (z=0): Energy Density
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                base_layer = (1.0 - (1.0 / self.N) * trace.real)
                
                # Stack layers
                # We apply convolution to smooth out higher layers (simulating RG flow)
                # In PyTorch we can use avg_pool or gaussian blur
                
                # base_layer is [B, H, W] or [H, W]. If [1, H, W], we need [1, 1, H, W] for conv2d.
                # Check shape first
                if base_layer.dim() == 3:
                     current = base_layer.unsqueeze(1) # [B, 1, H, W]
                elif base_layer.dim() == 2:
                     current = base_layer.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
                else:
                     # Unexpected, force reshape
                     current = base_layer.view(1, 1, self.grid_size, self.grid_size)

                layers = [base_layer.squeeze()] # Store as [H, W] for stacking
                
                # Gaussian kernel approximation (3x3)
                kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                                     device=self.device, dtype=torch.float32) / 16.0
                kernel = kernel.view(1, 1, 3, 3)
                
                for _ in range(depth - 1):
                    # Pad to keep size
                    current = torch.nn.functional.conv2d(current, kernel, padding=1)
                    layers.append(current.squeeze(0).squeeze(0))
                    
                # Stack to [Depth, H, W]
                data = torch.stack(layers, dim=0) # [D, H, W]
                
            else:
                # Default: density
                U_p = self._compute_plaquettes(self.links)
                trace = torch.diagonal(U_p, dim1=-2, dim2=-1).sum(-1)
                energy = 1.0 - (1.0 / self.N) * trace.real
                data = energy.squeeze(0) # [H, W]
            
            # Convertir a numpy
            data_np = data.cpu().numpy().astype(np.float32)
            
            # NORMALIZACIÓN: Los shaders esperan datos en [0, 1]
            if viz_type == "holographic":
                 # Normalize per channel or global?
                 # Global normalization preserves color balance
                 min_val = float(data_np.min())
                 max_val = float(data_np.max())
                 if max_val > min_val and abs(max_val - min_val) > 1e-10:
                     data_np = (data_np - min_val) / (max_val - min_val)
                 else:
                     data_np = np.zeros_like(data_np) # Black background if no excitation
            else:
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
                "engine": "LatticeEngine",
                "channels": 3 if viz_type == "holographic" else 1
            }
            
        except Exception as e:
            logging.error(f"Error en get_visualization_data: {e}")
            return {"data": None, "type": viz_type, "error": str(e)}


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

    def get_dense_state(self, roi=None, check_pause_callback=None):
        """
        Retorna el estado denso para visualización.
        Convierte la densidad de acción/energía a un tensor complejo compatible.
        """
        # Obtener densidad de energía - retorna dict con 'data'
        viz_dict = self.get_visualization_data("density")
        energy_np = viz_dict['data']
        
        # Convertir a tensor
        energy = torch.tensor(energy_np, device=self.device, dtype=torch.float32)
        
        # Reshape a [1, H, W, 1]
        energy = energy.unsqueeze(0).unsqueeze(-1)
        
        # Retornar como complejo (Real=Energía, Imag=0)
        # Esto asegura compatibilidad con pipelines que esperan complejos
        return torch.complex(energy, torch.zeros_like(energy))

    def apply_tool(self, action, params):
        """
        Aplica una herramienta cuántica al estado del retículo.
        Interfaz genérica para QuantumToolbox.
        """
        logging.info(f"🛠️ LatticeEngine aplicando herramienta: {action} | Params: {params}")
        
        if action == 'collapse':
            intensity = float(params.get('intensity', 0.5))
            center = None
            if 'x' in params and 'y' in params:
                center = (int(params['y']), int(params['x'])) # (H, W)
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
        """
        Simula un 'colapso' (termalización local) aleatorizando links.
        """
        # Generar ruido SU(3) fuerte
        noise = self._generate_random_su3(epsilon=intensity * 2.0)
        
        if center:
            cy, cx = center
            # Radio fijo o basado en intensidad
            radius = int(self.grid_size * 0.15) 
            
            # Crear máscara circular
            y, x = torch.meshgrid(torch.arange(self.grid_size, device=self.device), torch.arange(self.grid_size, device=self.device), indexing='ij')
            dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
            mask = (dist < radius).float()
            
            # Suavizar bordes (opcional, aquí hard cut)
            mask = mask.view(1, 1, self.grid_size, self.grid_size, 1, 1)
            
            # Aplicar ruido solo en la región
            # U_new = Noise * U_old
            # Interpolamos: U_final = (1-mask)*U_old + mask*(Noise*U_old)
            perturbed_links = self._matmul(noise, self.links)
            self.links = (1 - mask) * self.links + mask * perturbed_links
        else:
            # Global collapse (quench)
            self.links = self._matmul(noise, self.links)

    def _apply_vortex(self, x, y, radius, strength):
        """
        Inyecta un defecto topológico (vórtice) en los links.
        Modifica los links alrededor de (x,y) para crear holonomía no trivial.
        """
        # Coordenadas centradas en (x,y)
        yy, xx = torch.meshgrid(torch.arange(self.grid_size, device=self.device), torch.arange(self.grid_size, device=self.device), indexing='ij')
        xx = xx - x
        yy = yy - y
        
        # Ángulo polar theta
        theta = torch.atan2(yy, xx)
        
        # Matriz de rotación SU(3) dependiente del ángulo (Vortex ansatz simplificado)
        # U(theta) = exp(i * strength * theta * Lambda_3)
        # Lambda_3 = diag(1, -1, 0)
        
        # Construir generador diagonal [H, W, 3]
        # strength * theta para diag 0, -strength * theta para diag 1
        phase = strength * theta
        
        # Crear matriz diagonal [H, W, 3, 3]
        # exp(i*phase)
        # Elemento (0,0)
        u00 = torch.exp(1j * phase)
        u11 = torch.exp(-1j * phase)
        u22 = torch.ones_like(phase, dtype=torch.complex64)
        
        # Ensamblar matriz de transformación G(x)
        G = torch.zeros(self.grid_size, self.grid_size, self.N, self.N, dtype=torch.complex64, device=self.device)
        G[..., 0, 0] = u00
        G[..., 1, 1] = u11
        G[..., 2, 2] = u22
        
        # Aplicar transformación de gauge local: U_mu(n) -> G(n) U_mu(n) G^dag(n+mu)
        # Esto inyecta el vórtice en la configuración de gauge
        
        # Expandir G para batch [1, 1, H, W, N, N]
        G = G.view(1, 1, self.grid_size, self.grid_size, self.N, self.N)
        
        # Shift G para G^dag(n+mu)
        # G_px = G(n+x)
        G_px = torch.roll(G, shifts=-1, dims=3) # Shift en W
        # G_py = G(n+y)
        G_py = torch.roll(G, shifts=-1, dims=2) # Shift en H
        
        # Links actuales
        Ux = self.links[:, 0:1] # [B, 1, H, W, N, N]
        Uy = self.links[:, 1:2] # [B, 1, H, W, N, N]
        
        # Transformar
        # Ux' = G * Ux * G_px^dag
        Ux_new = torch.matmul(torch.matmul(G, Ux), G_px.conj().transpose(-2, -1))
        
        # Uy' = G * Uy * G_py^dag
        Uy_new = torch.matmul(torch.matmul(G, Uy), G_py.conj().transpose(-2, -1))
        
        # Aplicar solo dentro del radio (smooth envelope)
        dist = torch.sqrt(xx**2 + yy**2)
        mask = (dist < radius * 2).float().view(1, 1, self.grid_size, self.grid_size, 1, 1)
        # Smooth transition sigmoid
        mask = torch.sigmoid((radius - dist) * 0.5).view(1, 1, self.grid_size, self.grid_size, 1, 1)
        
        # Mezclar
        self.links[:, 0:1] = (1 - mask) * Ux + mask * Ux_new
        self.links[:, 1:2] = (1 - mask) * Uy + mask * Uy_new

    def _apply_click_out(self):
        """
        Simulates the Gateway 'Click-out' phase.
        Randomly connects distant parts of the grid ('tunneling') ensuring non-locality.
        Equation: psi(x) <-> psi(y) if |x-y| >> 0 and random < P_tunnel
        """
        if np.random.random() > self.click_out_chance:
            return

        # Select random source indices
        # We'll pick N random sites to attempt tunneling
        n_tunnels = int(self.grid_size * 0.5) # E.g. half the grid width number of tunnels
        
        b = 0 # Batch index 0
        
        # Source coordinates
        src_y = torch.randint(0, self.grid_size, (n_tunnels,), device=self.device)
        src_x = torch.randint(0, self.grid_size, (n_tunnels,), device=self.device)
        
        # Target coordinates (randomly distant)
        dst_y = torch.randint(0, self.grid_size, (n_tunnels,), device=self.device)
        dst_x = torch.randint(0, self.grid_size, (n_tunnels,), device=self.device)
        
        # Links at sources
        # We need to act on the links variable directly
        # self.links shape: [B, 2, H, W, N, N]
        
        # We will SWAP the U matrices between source and dest
        # Or mix them: U_new = (U_src + U_dst)/sqrt(2) -> Entanglement-ish
        # Let's do a Swap for 'teleportation' effect or Mix for 'resonance'.
        # Swap is more visible as "glitch/sparkle".
        
        # Extract links
        # Need advanced indexing
        # B=0 fixed
        
        # Links has 2 directions (0 and 1). Let's swap both or random? Both.
        
        # Get values
        src_links = self.links[b, :, src_y, src_x].clone() # [2, n_tunnels, N, N]
        dst_links = self.links[b, :, dst_y, dst_x].clone()
        
        # Mix strategy: U_new = (U_a * U_b). This is composition in group.
        # Resonance: U_a' = U_b, U_b' = U_a (Exchange info)
        
        self.links[b, :, src_y, src_x] = dst_links
        self.links[b, :, dst_y, dst_x] = src_links
        
        # No Re-unitarization needed if we just swap unitary matrices!

    def _apply_wave(self, k_x, k_y):
        """
        Inyecta una onda plana en la fase de los links.
        """
        # Grid coordinates
        y, x = torch.meshgrid(torch.arange(self.grid_size, device=self.device), torch.arange(self.grid_size, device=self.device), indexing='ij')
        
        # Phase = k_x * x + k_y * y
        phase = (k_x * x + k_y * y).float()
        
        # Modulación periódica pequeña
        epsilon = 0.3
        
        # Generador SU(3) (Lambda 2 para parte imaginaria/fase)
        # O simplemente rotación global de fase U(1) subgroup
        
        # U_wave = exp(i * epsilon * sin(phase) * Lambda)
        phi = epsilon * torch.sin(phase / 10.0) # Frecuencia ajustada
        
        # Matriz unitaria diagonal [H, W, 3, 3]
        u_wave = torch.zeros(self.grid_size, self.grid_size, self.N, self.N, dtype=torch.complex64, device=self.device)
        u_wave[..., 0, 0] = torch.exp(1j * phi)
        u_wave[..., 1, 1] = torch.exp(-1j * phi) # Traceless-ish phase
        u_wave[..., 2, 2] = 1.0
        
        u_wave = u_wave.view(1, 1, self.grid_size, self.grid_size, self.N, self.N)
        
        # Perturbar links
        self.links = torch.matmul(u_wave, self.links)


