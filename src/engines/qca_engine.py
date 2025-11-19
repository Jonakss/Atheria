# src/qca_engine.py
import torch
import torch.nn as nn
import numpy as np
import os
import logging
from . import config as cfg

class QuantumState:
    """
    Estado cuántico con soporte para memoria temporal (ConvLSTM).
    """
    def __init__(self, grid_size, d_state, device, d_memory=None, initial_mode='complex_noise'):
        self.grid_size = grid_size
        self.d_state = d_state
        self.device = device
        self.d_memory = d_memory  # Dimensión de memoria para ConvLSTM
        self.psi = self._initialize_state(mode=initial_mode)
        # Estados de memoria para ConvLSTM (h_state y c_state)
        # Se inicializarán cuando se use un modelo ConvLSTM
        self.h_state = None
        self.c_state = None
        
    def _initialize_state(self, mode='complex_noise', complex_noise_strength=0.1):
        if mode == 'random':
            real = torch.randn(1, self.grid_size, self.grid_size, self.d_state, device=self.device)
            imag = torch.randn(1, self.grid_size, self.grid_size, self.d_state, device=self.device)
            psi_complex = torch.complex(real, imag)
            norm = torch.sqrt(torch.sum(psi_complex.abs().pow(2), dim=-1, keepdim=True))
            return psi_complex / norm
        elif mode == 'complex_noise':
            noise = torch.randn(1, self.grid_size, self.grid_size, self.d_state, device=self.device) * complex_noise_strength
            real, imag = torch.cos(noise), torch.sin(noise)
            return torch.complex(real, imag)
        else:
            return torch.zeros(1, self.grid_size, self.grid_size, self.d_state, device=self.device, dtype=torch.complex64)
    
    def _reset_state_random(self): 
        self.psi = self._initialize_state(mode='random')
        # Resetear memoria si existe
        if self.h_state is not None:
            self.h_state.zero_()
        if self.c_state is not None:
            self.c_state.zero_()
    
    def initialize_memory(self, memory_shape):
        """
        Inicializa los estados de memoria h_state y c_state para ConvLSTM.
        
        Args:
            memory_shape: Tupla (batch, channels, H, W) para los estados de memoria
        """
        self.h_state = torch.zeros(1, *memory_shape, device=self.device, dtype=torch.float32)
        self.c_state = torch.zeros(1, *memory_shape, device=self.device, dtype=torch.float32)
    
    def load_state(self, filepath):
        try:
            state_dict = torch.load(filepath, map_location=self.device)
            self.psi = state_dict['psi'].to(self.device)
            # Cargar memoria si existe
            if 'h_state' in state_dict:
                self.h_state = state_dict['h_state'].to(self.device)
            if 'c_state' in state_dict:
                self.c_state = state_dict['c_state'].to(self.device)
        except Exception: 
            self._reset_state_random()
    
    def save_state(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_dict = {'psi': self.psi.cpu()}
        if self.h_state is not None:
            save_dict['h_state'] = self.h_state.cpu()
        if self.c_state is not None:
            save_dict['c_state'] = self.c_state.cpu()
        torch.save(save_dict, filepath)

class Aetheria_Motor:
    def __init__(self, model_operator: nn.Module, grid_size: int, d_state: int, device, cfg=None, d_memory=None):
        self.device = device
        self.grid_size = int(grid_size)
        self.d_state = int(d_state)
        
        # Guardar referencia al modelo original antes de cualquier optimización
        self.original_model = model_operator
        
        # Optimizar modelo para inferencia
        from .gpu_optimizer import get_optimizer
        self.optimizer = get_optimizer(device)
        model_operator = self.optimizer.optimize_model(model_operator)
        
        self.operator = model_operator.to(self.device)
        
        # Detectar si el modelo usa ConvLSTM (usar modelo original para detección)
        self.has_memory = hasattr(self.original_model, 'convlstm') or 'ConvLSTM' in self.original_model.__class__.__name__
        
        # Obtener modo de inicialización desde cfg o usar default
        initial_mode = 'complex_noise'
        if cfg is not None:
            initial_mode = getattr(cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
        
        # Inicializar estado cuántico con soporte para memoria si es necesario
        self.state = QuantumState(self.grid_size, self.d_state, device, d_memory=d_memory, initial_mode=initial_mode)
        
        self.is_compiled = False
        self.cfg = cfg
        # Almacenar delta_psi y psi_input para visualizaciones
        self.last_delta_psi = None
        self.last_psi_input = None  # Guardar psi de entrada para cálculo de matriz A
        self.last_delta_psi_decay = None  # Para visualizar el término Lindbladian
        
        if self.has_memory:
            logging.info(f"Motor con memoria temporal (ConvLSTM) inicializado. d_memory={d_memory}")

    def evolve_internal_state(self):
        if self.state.psi is None: return
        
        # Usar inference_mode en lugar de no_grad para mejor rendimiento
        from .gpu_optimizer import GPUOptimizer
        with GPUOptimizer.enable_inference_mode():
            # Limpiar cache de GPU periódicamente
            self.optimizer.empty_cache_if_needed()
            
            self.state.psi = self._evolve_logic(self.state.psi)

    def evolve_step(self, current_psi):
        with torch.set_grad_enabled(True):
            return self._evolve_logic(current_psi)

    def _evolve_logic(self, psi_in):
        """
        Implementa la Ecuación Maestra de Lindblad para sistemas cuánticos abiertos.
        
        dρ/dt = -i[H, ρ] + Σ_i γ_i (L_i ρ L_i† - (1/2){L_i† L_i, ρ})
        
        Para nuestro caso simplificado (disipación simple):
        - Parte 1 (Unitaria): dΨ/dt = A(Ψ) * Ψ (implementada por la Ley M / U-Net)
        - Parte 2 (Lindbladian): dΨ/dt = -γ * Ψ (decaimiento/disipación)
        
        El modelo debe aprender a "bombear" energía contra el decaimiento para mantener
        estructuras estables (A-Life / metabolismo).
        """
        # Guardar psi_in para cálculos posteriores
        self.last_psi_input = psi_in
        
        # ====================================================================
        # PARTE 1: Evolución Unitaria (Sistema Cerrado)
        # dΨ/dt = A(Ψ) * Ψ
        # Esta es la "Ley M" aprendida por la U-Net (o U-Net con ConvLSTM)
        # ====================================================================
        x_cat_real = psi_in.real.permute(0, 3, 1, 2)
        x_cat_imag = psi_in.imag.permute(0, 3, 1, 2)
        x_cat_total = torch.cat([x_cat_real, x_cat_imag], dim=1)
        
        # Si el modelo tiene memoria (ConvLSTM), pasar estados de memoria
        if self.has_memory:
            # Obtener estados de memoria del estado cuántico
            # Los estados pueden ser None (primera vez) o tener forma [1, batch, channels, H, W]
            h_t = self.state.h_state if self.state.h_state is not None else None
            c_t = self.state.c_state if self.state.c_state is not None else None
            
            # Llamar al modelo con memoria
            # IMPORTANTE: Detach h_t y c_t antes de pasarlos al modelo para evitar problemas con backward
            if h_t is not None:
                h_t = h_t.detach()
            if c_t is not None:
                c_t = c_t.detach()
            
            delta_psi_unitario_complex, h_next, c_next = self.operator(x_cat_total, h_t, c_t)
            
            # Guardar nuevos estados de memoria (ya tienen la forma correcta [1, batch, channels, H, W])
            # IMPORTANTE: Detach antes de guardar para evitar problemas con backward en el siguiente episodio
            self.state.h_state = h_next.detach() if h_next is not None else None
            self.state.c_state = c_next.detach() if c_next is not None else None
        else:
            # Modelo sin memoria (U-Net estándar)
            delta_psi_unitario_complex = self.operator(x_cat_total)

        delta_real, delta_imag = torch.chunk(delta_psi_unitario_complex, 2, dim=1)
        delta_psi_unitario = torch.complex(delta_real, delta_imag).permute(0, 2, 3, 1)
        
        # Para UNetUnitary, aplicar transformación unitaria explícita
        model_name = self.operator.__class__.__name__
        if model_name == 'UNetUnitary':
            # Multiplicación matricial compleja: A * Ψ
            A_times_psi_real = delta_psi_unitario.real * psi_in.real - delta_psi_unitario.imag * psi_in.imag
            A_times_psi_imag = delta_psi_unitario.real * psi_in.imag + delta_psi_unitario.imag * psi_in.real
            delta_psi_unitario = torch.complex(A_times_psi_real, A_times_psi_imag)

        # ====================================================================
        # PARTE 2: Término Lindbladian (Sistema Abierto)
        # dΨ/dt = -γ * Ψ
        # Esto implementa la "disipación" o "decaimiento"
        # Representa la interacción con el entorno (pérdida de energía/coherencia)
        # ====================================================================
        # Obtener GAMMA_DECAY desde cfg (puede estar en exp_cfg o global_cfg)
        gamma_decay = 0.0
        if self.cfg is not None:
            if hasattr(self.cfg, 'GAMMA_DECAY'):
                gamma_decay = self.cfg.GAMMA_DECAY
            elif isinstance(self.cfg, dict) and 'GAMMA_DECAY' in self.cfg:
                gamma_decay = self.cfg['GAMMA_DECAY']
        
        delta_psi_decay = None
        if gamma_decay > 0:
            # Término Lindbladian: decaimiento exponencial hacia el vacío
            # Esto es equivalente al operador de aniquilación L = a (disipación)
            delta_psi_decay = -gamma_decay * psi_in
            self.last_delta_psi_decay = delta_psi_decay.clone()
            
            # Combinar evolución unitaria y decaimiento
            # dΨ/dt = [A(Ψ) * Ψ] + [-γ * Ψ]
            # La Ley M debe "ganar" contra el decaimiento para mantener estructuras
            delta_psi_total = delta_psi_unitario + delta_psi_decay
        else:
            # Sistema cerrado puro (sin decaimiento)
            delta_psi_total = delta_psi_unitario
            self.last_delta_psi_decay = None
        
        # ====================================================================
        # Aplicar Método de Euler
        # Ψ(t+1) = Ψ(t) + dΨ/dt * dt (con dt=1)
        # ====================================================================
        new_psi = psi_in + delta_psi_total
        
        # Normalizar para mantener la norma del estado cuántico
        # (Esto es necesario porque la combinación unitaria + decaimiento puede cambiar la norma)
        norm = torch.sqrt(torch.sum(new_psi.abs().pow(2), dim=-1, keepdim=True))
        new_psi = new_psi / (norm + 1e-9)
        
        # Guardar delta_psi unitario para visualizaciones (quiver plot)
        # (No guardamos delta_psi_total porque queremos ver solo la "fuerza de vida")
        self.last_delta_psi = delta_psi_unitario.clone()
        
        return new_psi
    
    def get_physics_matrix(self, x: int, y: int):
        """
        Obtiene la matriz de física local A para una célula específica.
        Calcula la matriz A tal que delta_psi ≈ A @ psi usando un método numérico mejorado.
        
        Args:
            x, y: Coordenadas de la célula
        
        Returns:
            numpy array de shape (d_state, d_state) con la matriz A (compleja)
        """
        if self.last_psi_input is None or self.last_delta_psi is None:
            return None
        
        try:
            # Obtener psi y delta_psi para la célula (x, y)
            psi_cell = self.last_psi_input[0, y, x, :].cpu().numpy()  # shape: (d_state,) complejo
            delta_psi_cell = self.last_delta_psi[0, y, x, :].cpu().numpy()  # shape: (d_state,) complejo
            
            d_state = self.d_state
            
            # Método mejorado: calcular A usando múltiples perturbaciones
            # Para cada canal j, calcular cómo afecta psi[j] a delta_psi[i]
            A_matrix = np.zeros((d_state, d_state), dtype=np.complex128)
            
            # Si psi es muy pequeño, usar aproximación directa
            psi_norm = np.abs(psi_cell).sum()
            if psi_norm < 1e-10:
                # Estado muy pequeño, usar aproximación lineal simple
                for i in range(d_state):
                    for j in range(d_state):
                        # Aproximación: A[i,j] ≈ delta_psi[i] / psi[j] si psi[j] != 0
                        if abs(psi_cell[j]) > 1e-10:
                            A_matrix[i, j] = delta_psi_cell[i] / psi_cell[j]
            else:
                # Método más preciso: usar relación compleja
                # delta_psi = A @ psi, entonces A = delta_psi @ pinv(psi)
                # Para matrices complejas, usamos pseudoinversa de Moore-Penrose
                psi_vec = psi_cell.reshape(-1, 1)  # (d_state, 1)
                delta_psi_vec = delta_psi_cell.reshape(-1, 1)  # (d_state, 1)
                
                # Calcular pseudoinversa de psi
                # Para un vector, pinv(psi) = psi* / |psi|²
                psi_conj = np.conj(psi_vec)  # (d_state, 1)
                psi_norm_sq = np.real(np.dot(psi_conj.T, psi_vec))[0, 0]
                
                if psi_norm_sq > 1e-10:
                    psi_pinv = psi_conj / psi_norm_sq  # (d_state, 1)
                    # A = delta_psi @ pinv(psi) = delta_psi @ (psi* / |psi|²)
                    A_matrix = np.dot(delta_psi_vec, psi_pinv.T)  # (d_state, d_state)
                else:
                    # Fallback: usar aproximación elemento a elemento
                    for i in range(d_state):
                        for j in range(d_state):
                            if abs(psi_cell[j]) > 1e-10:
                                A_matrix[i, j] = delta_psi_cell[i] / psi_cell[j]
            
            return A_matrix
        except Exception as e:
            logging.warning(f"Error calculando matriz de física en ({x}, {y}): {e}")
            return None
    
    def get_physics_matrix_map(self, sample_rate=1):
        """
        Calcula un mapa de la "fuerza física" para todas las células.
        Usa una métrica agregada de la matriz A (magnitud promedio, traza, etc.)
        
        Args:
            sample_rate: Calcular cada N células (1 = todas, 2 = cada 2, etc.) para optimizar
        
        Returns:
            numpy array de shape (H, W) con métricas de física por célula
        """
        if self.last_psi_input is None or self.last_delta_psi is None:
            return None
        
        try:
            H, W = self.grid_size, self.grid_size
            physics_map = np.zeros((H, W), dtype=np.float32)
            
            # Optimización: calcular directamente desde tensores en lugar de llamar get_physics_matrix
            # para cada célula individualmente
            psi_np = self.last_psi_input[0].cpu().numpy()  # (H, W, d_state) complejo
            delta_psi_np = self.last_delta_psi[0].cpu().numpy()  # (H, W, d_state) complejo
            
            # Calcular métrica de física para cada célula
            # Métrica: magnitud promedio de la relación delta_psi / psi
            # Esto aproxima la "fuerza" de la interacción física local
            for y in range(0, H, sample_rate):
                for x in range(0, W, sample_rate):
                    psi_cell = psi_np[y, x, :]  # (d_state,)
                    delta_psi_cell = delta_psi_np[y, x, :]  # (d_state,)
                    
                    # Calcular magnitud de la relación delta_psi / psi
                    # Evitar división por cero
                    psi_magnitude = np.abs(psi_cell)
                    delta_psi_magnitude = np.abs(delta_psi_cell)
                    
                    # Métrica: promedio de |delta_psi| / (|psi| + epsilon)
                    # Esto mide la "fuerza" de la transformación física
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = np.where(psi_magnitude > 1e-10, 
                                        delta_psi_magnitude / (psi_magnitude + 1e-10),
                                        0.0)
                    physics_map[y, x] = np.mean(ratio)
            
            # Interpolar valores para células no calculadas si sample_rate > 1
            if sample_rate > 1:
                # Interpolación simple usando numpy
                # Repetir valores para llenar los huecos (interpolación nearest neighbor)
                expanded_map = np.zeros((H, W), dtype=np.float32)
                calc_h, calc_w = physics_map.shape[0], physics_map.shape[1]
                for y in range(H):
                    for x in range(W):
                        src_y = min(int(y * calc_h / H), calc_h - 1)
                        src_x = min(int(x * calc_w / W), calc_w - 1)
                        expanded_map[y, x] = physics_map[src_y, src_x]
                physics_map = expanded_map
            
            return physics_map
        except Exception as e:
            import logging
            logging.warning(f"Error calculando mapa de física: {e}")
            # Fallback: usar aproximación más simple
            try:
                delta_psi_magnitude = np.abs(self.last_delta_psi[0].cpu().numpy())
                return np.mean(delta_psi_magnitude, axis=-1)  # (H, W)
            except:
                return None

    def propagate(self, psi_inicial, num_steps):
        psi_history = []
        psi_actual = psi_inicial
        for _ in range(num_steps):
            psi_actual = self.evolve_step(psi_actual)
            psi_history.append(psi_actual)
        return psi_history, psi_actual
    
    def get_initial_state(self, batch_size: int):
        real = torch.randn(batch_size, self.grid_size, self.grid_size, self.d_state, device=self.device)
        imag = torch.randn(batch_size, self.grid_size, self.grid_size, self.d_state, device=self.device)
        psi_complex = torch.complex(real, imag)
        norm = torch.sqrt(torch.sum(psi_complex.abs().pow(2), dim=-1, keepdim=True))
        return psi_complex / (norm + 1e-9)

    def laplacian_2d_psi(self, psi):
        psi_permuted = psi.permute(0, 3, 1, 2)
        d_state = psi_permuted.shape[1]
        kernel_base = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                   dtype=torch.float32, device=self.device).reshape(1, 1, 3, 3)
        kernel = kernel_base.repeat(d_state, 1, 1, 1)
        laplacian_real = nn.functional.conv2d(psi_permuted.real, kernel, padding=1, groups=d_state)
        laplacian_imag = nn.functional.conv2d(psi_permuted.imag, kernel, padding=1, groups=d_state)
        laplacian_complex = torch.complex(laplacian_real, laplacian_imag)
        return laplacian_complex.permute(0, 2, 3, 1)

    def compile_model(self):
        if not hasattr(self.original_model, '_compiles') or self.original_model._compiles:
            if not self.is_compiled:
                try:
                    logging.info("Aplicando torch.compile() al modelo...")
                    self.operator = torch.compile(self.operator, mode="reduce-overhead")
                    self.is_compiled = True
                    logging.info("¡torch.compile() aplicado exitosamente!")
                except Exception as e:
                    logging.warning(f"torch.compile() falló: {e}. El modelo se ejecutará sin compilar.")
        else:
            logging.info(f"torch.compile() omitido para el modelo {self.original_model.__class__.__name__} según su configuración.")
    
    def get_model_for_params(self):
        """
        Obtiene el modelo que se puede usar para acceder a parámetros.
        Si el modelo está compilado, devuelve el modelo original.
        """
        # Si está compilado, intentar acceder al modelo original
        if self.is_compiled:
            # torch.compile puede envolver el modelo, intentar acceder a _orig_mod
            if hasattr(self.operator, '_orig_mod'):
                return self.operator._orig_mod
            # Si no tiene _orig_mod, usar el modelo original guardado
            return self.original_model
        # Si no está compilado, devolver el operador directamente
        return self.operator