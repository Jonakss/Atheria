# src/qca_engine.py
import torch
import torch.nn as nn
import numpy as np
import os
import logging
from .. import config as cfg
from src.cache import cache

# Versión del motor Python
try:
    from .__version__ import __version__ as ENGINE_VERSION
except ImportError:
    ENGINE_VERSION = "4.1.0"  # Fallback

class QuantumState:
    """
    Estado cuántico con soporte para memoria temporal (ConvLSTM).
    """
    def __init__(self, grid_size, d_state, device, d_memory=None, initial_mode='complex_noise', base_state=None, base_grid_size=None, precomputed_state=None):
        self.grid_size = grid_size
        self.d_state = d_state
        self.device = device
        self.d_memory = d_memory  # Dimensión de memoria para ConvLSTM
        self.psi = self._initialize_state(mode=initial_mode, base_state=base_state, base_grid_size=base_grid_size, precomputed_state=precomputed_state)
        # Estados de memoria para ConvLSTM (h_state y c_state)
        # Se inicializarán cuando se use un modelo ConvLSTM
        self.h_state = None
        self.c_state = None
        
    def _initialize_state(self, mode='complex_noise', complex_noise_strength=0.1, base_state=None, base_grid_size=None, precomputed_state=None):
        """
        Inicializa el estado cuántico.
        
        Args:
            mode: Modo de inicialización ('random', 'complex_noise', 'zeros')
            complex_noise_strength: Intensidad del ruido complejo
            base_state: Estado base opcional a replicar (tensor complejo de tamaño menor)
            base_grid_size: Tamaño del grid del estado base (si se proporciona base_state)
            precomputed_state: Tensor de estado ya calculado (para inyección directa)
        
        Si se proporciona base_state y base_grid_size < self.grid_size,
        el estado base se replicará (tile) en el grid más grande.
        """
        # Si se proporciona un estado precalculado, usarlo directamente
        if precomputed_state is not None:
            return precomputed_state.to(self.device)

        # Si hay un estado base y el grid actual es más grande, replicar (tile) el estado
        if base_state is not None and base_grid_size is not None and base_grid_size < self.grid_size:
            return self._replicate_state(base_state, base_grid_size, self.grid_size)
        
        # Inicialización normal
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
        elif mode == 'ionq':
            return self._get_ionq_state(complex_noise_strength)
        else:
            return torch.zeros(1, self.grid_size, self.grid_size, self.d_state, device=self.device, dtype=torch.complex64)

    def _get_ionq_state(self, strength=0.1):
        """
        Generates an initial state using the IonQ Quantum Computer.
        "Quantum Genesis": The universe starts from true quantum superposition/entanglement.
        """
        logging.info("⚛️ Quantum Genesis: Initializing universe from IonQ...")
        try:
            # Lazy import to avoid circular dependencies/errors if not used
            from .compute_backend import IonQBackend
            from .. import config as cfg
            
            # Initialize Backend
            backend = IonQBackend(api_key=cfg.IONQ_API_KEY, backend_name=cfg.IONQ_BACKEND_NAME)
            
            # Create a Quantum Circuit (Superposition + Entanglement)
            # We use a simple circuit to generate complex correlations
            from qiskit import QuantumCircuit
            n_qubits = 11 # Standard for IonQ basic access
            qc = QuantumCircuit(n_qubits)
            
            # 1. Superposition (Hadamard on all)
            qc.h(range(n_qubits))
            
            # 2. Entanglement (CNOT chain)
            for i in range(n_qubits - 1):
                qc.cx(i, i+1)
            qc.cx(n_qubits-1, 0) # Close the loop
            
            # 3. Measurement
            qc.measure_all()
            
            # Execute
            # We request enough shots to fill a reasonable portion of the grid
            # Cost Optimization: Use the bitstrings from shots to fill the grid
            # 1024 shots * 11 bits = ~11k bits. 
            # Grid 256x256 = 65k cells. We will tile/repeat.
            shots = 1024 
            counts = backend.execute('run_circuit', qc, shots=shots)
            
            # Convert counts to a tensor
            # We'll construct a noise tensor from the bitstrings
            # This is a simple mapping: '0' -> -1, '1' -> +1
            
            # Flatten results into a long string of bits
            bit_stream = []
            for bitstring, count in counts.items():
                # Repeat bitstring 'count' times to respect probability distribution
                # or just use unique bitstrings? 
                # For "noise", respecting distribution is better representation of the wavefunction
                bits = [1.0 if c == '1' else -1.0 for c in bitstring]
                bit_stream.extend(bits * count)
                
            # Convert to tensor
            quantum_data = torch.tensor(bit_stream, device=self.device, dtype=torch.float32)
            
            # Reshape/Resize to match grid
            total_needed = self.grid_size * self.grid_size * self.d_state
            
            # Repeat the stream to fill the grid
            repeats = (total_needed // len(quantum_data)) + 1
            quantum_data = quantum_data.repeat(repeats)[:total_needed]
            
            # Reshape to (1, H, W, d_state)
            noise = quantum_data.reshape(1, self.grid_size, self.grid_size, self.d_state)
            
            # Scale by strength
            noise = noise * strength
            
            # Map to complex phase
            real, imag = torch.cos(noise), torch.sin(noise)
            logging.info("✨ Quantum Genesis Complete.")
            return torch.complex(real, imag)
            
        except Exception as e:
            logging.error(f"❌ Quantum Genesis Failed: {e}. Falling back to pseudo-random noise.")
            return self._initialize_state(mode='complex_noise', complex_noise_strength=strength)
    
    @staticmethod
    def create_variational_state(grid_size, d_state, device, params, strength=0.1, backend_name=None):
        """
        Generates a state using a Variational Quantum Circuit (VQC).
        Used by Quantum Tuner to optimize initialization.
        
        Args:
            params (list): List of rotation angles [theta_0, theta_1, ...]
            backend_name (str): Name of backend to use (optional, defaults to config or local_aer if failed)
        """
        import logging
        from .backend_factory import BackendFactory
        from qiskit import QuantumCircuit
        
        logging.info(f"⚛️ Variational Genesis: Params={params[:3]}...")
        
        try:
            # Determine backend
            if backend_name:
                backend = BackendFactory.get_backend(backend_name)
            else:
                 backend = BackendFactory.get_backend('local_aer')

            n_qubits = 11
            qc = QuantumCircuit(n_qubits)
            
            # Variational Layer: Ry(theta) + Entanglement
            # We map params to qubits. If fewer params than qubits, recycle.
            for i in range(n_qubits):
                theta = params[i % len(params)]
                qc.ry(theta, i)
                qc.rx(theta/2, i) # Add some complexity
                
            # Entanglement Ring
            for i in range(n_qubits - 1):
                qc.cx(i, i+1)
            qc.cx(n_qubits-1, 0)
            
            qc.measure_all()
            
            # Execute
            counts = backend.execute('run_circuit', qc, shots=1024)
            
            # Process to Tensor
            bit_stream = []
            for bitstring, count in counts.items():
                bits = [1.0 if c == '1' else -1.0 for c in bitstring]
                bit_stream.extend(bits * count)
                
            quantum_data = torch.tensor(bit_stream, device=device, dtype=torch.float32)
            total_needed = grid_size * grid_size * d_state
            repeats = (total_needed // len(quantum_data)) + 1
            quantum_data = quantum_data.repeat(repeats)[:total_needed]
            noise = quantum_data.reshape(1, grid_size, grid_size, d_state) * strength
            
            real, imag = torch.cos(noise), torch.sin(noise)
            return torch.complex(real, imag)
            
        except Exception as e:
            logging.error(f"❌ Variational Genesis Failed: {e}")
            # Fallback to random
            return QuantumState(grid_size, d_state, device, initial_mode='complex_noise').psi

    def _replicate_state(self, base_state, base_grid_size, target_grid_size):
        """
        Replica (tile) un estado de un grid más pequeño en un grid más grande.
        
        El estado base se repite múltiples veces para llenar el grid más grande.
        Esto mantiene la misma información pero en un espacio más grande.
        
        Args:
            base_state: Estado base [1, H_base, W_base, d_state]
            base_grid_size: Tamaño del grid base
            target_grid_size: Tamaño del grid destino
        
        Returns:
            Estado replicado [1, H_target, W_target, d_state]
        """
        # Calcular cuántas veces se debe repetir en cada dimensión
        reps_h = (target_grid_size + base_grid_size - 1) // base_grid_size  # Redondear hacia arriba
        reps_w = (target_grid_size + base_grid_size - 1) // base_grid_size
        
        # Repetir el estado base
        replicated = base_state.repeat(1, reps_h, reps_w, 1)
        
        # Cortar al tamaño exacto si es necesario
        if replicated.shape[1] > target_grid_size or replicated.shape[2] > target_grid_size:
            replicated = replicated[:, :target_grid_size, :target_grid_size, :]
        
        logging.info(f"🔄 Estado replicado: {base_grid_size}x{base_grid_size} → {target_grid_size}x{target_grid_size} (tiles: {reps_h}x{reps_w})")
        
        return replicated
    
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

class CartesianEngine:
    """
    Motor Python de Atheria 4 para simulaciones cuánticas.
    
    Versión: {ENGINE_VERSION}
    Sigue Semantic Versioning (SemVer): MAJOR.MINOR.PATCH
    """.format(ENGINE_VERSION=ENGINE_VERSION)
    
    # Versión del motor
    VERSION = ENGINE_VERSION
    
    def __init__(self, model_operator: nn.Module, grid_size: int, d_state: int, device, cfg=None, d_memory=None, initial_mode=None, precomputed_state=None):
        self.device = device
        self.grid_size = int(grid_size)
        self.d_state = int(d_state)
        self.operator = model_operator
        self.original_model = model_operator
        
        # Obtener modo de inicialización: argumento > cfg > default
        if initial_mode is None:
            initial_mode = 'complex_noise'
            if cfg is not None:
                initial_mode = getattr(cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
        
        # Si hay un training_grid_size diferente (menor) que inference_grid_size,
        # crear un estado base del tamaño de entrenamiento y replicarlo (tile) en el grid más grande
        base_state = None
        base_grid_size = None
        if cfg is not None:
            training_grid_size = getattr(cfg, 'GRID_SIZE_TRAINING', None)
            if training_grid_size and training_grid_size < self.grid_size:
                # Crear estado base del tamaño de entrenamiento
                base_state_temp = QuantumState(training_grid_size, self.d_state, device, initial_mode=initial_mode)
                base_state = base_state_temp.psi
                base_grid_size = training_grid_size
                logging.info(f"🔄 Grid escalado: Creando estado base {training_grid_size}x{training_grid_size} para replicar (tile) en {self.grid_size}x{self.grid_size}")
        
        # Detectar si el modelo tiene memoria (ConvLSTM)
        self.has_memory = hasattr(model_operator, 'convlstm') or 'ConvLSTM' in model_operator.__class__.__name__
        self.d_memory = d_memory
        
        # Inicializar estado cuántico con soporte para memoria si es necesario
        # Si hay base_state, se replicará automáticamente en _initialize_state
        self.state = QuantumState(self.grid_size, self.d_state, device, d_memory=d_memory, 
                                   initial_mode=initial_mode, base_state=base_state, base_grid_size=base_grid_size, precomputed_state=precomputed_state)
        
        self.is_compiled = False
        self.cfg = cfg
        # Almacenar delta_psi y psi_input para visualizaciones
        self.last_delta_psi = None
        self.last_psi_input = None  # Guardar psi de entrada para cálculo de matriz A
        self.last_delta_psi_decay = None  # Para visualizar el término Lindbladian
        
        if self.has_memory:
            logging.info(f"Motor con memoria temporal (ConvLSTM) inicializado. d_memory={d_memory}")
            
        # Inicializar optimizador GPU
        from ..optimization.gpu_optimizer import GPUOptimizer
        self.optimizer = GPUOptimizer(self.device)

    def evolve_internal_state(self, step=None):
        if self.state.psi is None: return
        
        # 1. Intentar recuperar de caché si tenemos el paso actual
        if step is not None and cache.enabled:
            cache_interval = getattr(self.cfg, 'CACHE_STATE_INTERVAL', 100) if self.cfg else 100
            
            if step % cache_interval == 0:
                exp_name = getattr(self.cfg, 'EXPERIMENT_NAME', 'default') if self.cfg else 'default'
                cache_key = f"state:{exp_name}:{step}"
                
                cached_state = cache.get(cache_key)
                if cached_state is not None:
                    try:
                        self.state.psi = torch.from_numpy(cached_state).to(self.device)
                        logging.debug(f"⚡ Cache HIT: Estado restaurado desde Dragonfly para paso {step}")
                        return
                    except Exception as e:
                        logging.warning(f"⚠️ Error restaurando estado de caché: {e}")
        
        # Usar inference_mode en lugar de no_grad para mejor rendimiento
        from ..optimization.gpu_optimizer import GPUOptimizer
        with GPUOptimizer.enable_inference_mode():
            self.optimizer.empty_cache_if_needed()
            self.state.psi = self._evolve_logic(self.state.psi)
            
        # 2. Guardar en caché si es intervalo
        if step is not None and cache.enabled:
            cache_interval = getattr(self.cfg, 'CACHE_STATE_INTERVAL', 100) if self.cfg else 100
            
            if step % cache_interval == 0:
                exp_name = getattr(self.cfg, 'EXPERIMENT_NAME', 'default') if self.cfg else 'default'
                cache_key = f"state:{exp_name}:{step}"
                
                try:
                    state_np = self.state.psi.cpu().numpy()
                    ttl = getattr(self.cfg, 'CACHE_TTL', 7200) if self.cfg else 7200
                    cache.set(cache_key, state_np, ttl=ttl)
                    logging.debug(f"💾 Estado guardado en Dragonfly para paso {step}")
                except Exception as e:
                    logging.warning(f"⚠️ Error guardando estado en caché: {e}")

    def evolve_step(self, current_psi):
        with torch.set_grad_enabled(True):
            return self._evolve_logic(current_psi)

    def evolve_hybrid_step(self, current_psi, step_num, injection_interval=10, noise_rate=0.05):
        """
        Evolución Híbrida: Ley M (Clásica/Neural) + Perturbación Cuántica (IonQ).
        
        Cada 'injection_interval' pasos, se inyecta ruido cuántico real para
        sacar al sistema de atractores cíclicos y fomentar la complejidad.
        """
        # 1. Evolución Normal
        new_psi = self.evolve_step(current_psi)
        
        # 2. Inyección Cuántica (Hybrid Compute)
        if step_num % injection_interval == 0:
            # Lazy init del inyector para no cargar dependencias si no se usa
            if not hasattr(self, 'ionq_collapse'):
                from ..physics.quantum_collapse import IonQCollapse
                self.ionq_collapse = IonQCollapse(self.device)
            
            # Inyectar Colapso IonQ
            # Elegimos un centro aleatorio para el evento de colapso
            # O podríamos usar una región de alta entropía (más "cuántica")
            import numpy as np
            cx = np.random.randint(0, self.grid_size)
            cy = np.random.randint(0, self.grid_size)
            
            new_psi = self.ionq_collapse.collapse(new_psi, region_center=(cy, cx), intensity=noise_rate)
            
            # También podemos mantener el ruido de fondo si se desea, 
            # pero el colapso es el evento principal aquí.
            
            import logging
            logging.info(f"⚡ Hybrid Event: Quantum Collapse at step {step_num}")
            
        return new_psi

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
            
            # Optimización Vectorizada: Calcular métrica para todo el grid a la vez
            # Evita bucles Python lentos (H*W iteraciones)
            
            # Calcular magnitudes
            psi_mag = np.abs(psi_np)
            delta_mag = np.abs(delta_psi_np)
            
            # Calcular ratio de forma segura (evitando división por cero)
            # ratio = |delta_psi| / (|psi| + epsilon)
            # Usar np.divide con where para seguridad extra, aunque epsilon suele bastar
            ratio = np.divide(delta_mag, psi_mag + 1e-10)
            
            # Si psi es muy pequeño (vacío), el ratio debería ser 0 (sin actividad física)
            ratio = np.where(psi_mag > 1e-10, ratio, 0.0)
            
            # Promediar sobre canales (d_state) para tener un valor por célula
            physics_map = np.mean(ratio, axis=-1)  # (H, W)
            
            # Aplicar subsampling si es necesario (para grids muy grandes)
            if sample_rate > 1:
                physics_map = physics_map[::sample_rate, ::sample_rate]
                
                # Si se necesita el tamaño original, interpolar (opcional, por ahora devolvemos subsampled)
                # El frontend suele manejar texturas de diferente resolución bien
                
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
                    # Silenciar temporalmente los mensajes de CUDA graph durante la compilación
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*cudagraph.*')
                        warnings.filterwarnings('ignore', message='.*CUDA graph.*')
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

    def apply_tool(self, action, params):
        """
        Aplica una herramienta cuántica al estado.
        """
        if self.state.psi is None:
            return False
            
        logging.info(f"🛠️ CartesianEngine aplicando herramienta: {action} | Params: {params}")
        
        try:
            # Lazy import de herramientas
            from ..physics import IonQCollapse, QuantumSteering
            
            device = self.device
            new_psi = None
            
            if action == 'collapse':
                intensity = float(params.get('intensity', 0.5))
                center = None
                if 'x' in params and 'y' in params:
                    center = (int(params['y']), int(params['x']))
                    
                collapser = IonQCollapse(device)
                new_psi = collapser.collapse(self.state.psi, region_center=center, intensity=intensity)
                
            elif action == 'vortex':
                x = int(params.get('x', self.grid_size // 2))
                y = int(params.get('y', self.grid_size // 2))
                radius = int(params.get('radius', 5))
                strength = float(params.get('strength', 1.0))
                
                steering = QuantumSteering(device)
                new_psi = steering.inject(self.state.psi, 'vortex', x=x, y=y, radius=radius, strength=strength)
                
            elif action == 'wave':
                k_x = float(params.get('k_x', 1.0))
                k_y = float(params.get('k_y', 1.0))
                
                # Onda global: aplicar en todo el grid
                # Usamos el centro del grid y un radio grande
                cx, cy = self.grid_size // 2, self.grid_size // 2
                radius = self.grid_size # Cubrir todo
                
                steering = QuantumSteering(device)
                new_psi = steering.inject(self.state.psi, 'plane_wave', x=cx, y=cy, radius=radius, k_x=k_x, k_y=k_y)
                
            else:
                logging.warning(f"⚠️ Herramienta no soportada: {action}")
                return False
    
            if new_psi is not None:
                self.state.psi = new_psi
                # Invalidar caché
                self.last_delta_psi = None
                return True
                
        except Exception as e:
            logging.error(f"❌ Error aplicando herramienta {action}: {e}", exc_info=True)
            return False

    def get_visualization_data(self, viz_type: str = "density"):
        """
        Retorna datos para visualización frontend.
        
        Args:
            viz_type: Tipo de visualización 
                - 'density': Magnitud/probabilidad |ψ|²
                - 'phase': Fase del campo cuántico
                - 'energy': Energía local
                - 'gradient': Gradiente de densidad
                - 'real': Parte real
                - 'imag': Parte imaginaria
                
        Returns:
            dict con 'data' (array 2D) y 'metadata'
        """
        if self.state.psi is None:
            return {"data": None, "type": viz_type, "error": "No state"}
        
        try:
            psi = self.state.psi  # [1, H, W, d_state]
            
            if viz_type == "density":
                # |ψ|² sumado sobre canales
                data = torch.sum(psi.abs().pow(2), dim=-1)[0]  # [H, W]
                
            elif viz_type == "phase":
                # Fase del primer canal (o promedio)
                phase = torch.angle(psi[..., 0])  # [1, H, W]
                data = phase[0]  # [H, W]
                
            elif viz_type == "energy":
                # Energía local (|∇ψ|² aproximado usando laplaciano)
                if self.last_delta_psi is not None:
                    data = torch.sum(self.last_delta_psi.abs().pow(2), dim=-1)[0]
                else:
                    data = torch.sum(psi.abs().pow(2), dim=-1)[0]
                    
            elif viz_type == "gradient":
                # Gradiente de densidad
                density = torch.sum(psi.abs().pow(2), dim=-1)  # [1, H, W]
                grad_x = torch.diff(density, dim=2)
                grad_y = torch.diff(density, dim=1)
                # Pad para mantener tamaño
                grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
                grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
                data = torch.sqrt(grad_x**2 + grad_y**2)[0]  # [H, W]
                
            elif viz_type == "real":
                # Parte real promediada sobre canales
                data = psi.real.mean(dim=-1)[0]  # [H, W]
                
            elif viz_type == "imag":
                # Parte imaginaria promediada
                data = psi.imag.mean(dim=-1)[0]  # [H, W]
                
            else:
                # Default: density
                data = torch.sum(psi.abs().pow(2), dim=-1)[0]
            
            # Convertir a numpy
            data_np = data.cpu().numpy().astype(np.float32)
            
            return {
                "data": data_np,
                "type": viz_type,
                "shape": list(data_np.shape),
                "min": float(data_np.min()),
                "max": float(data_np.max()),
                "engine": "CartesianEngine"
            }
            
        except Exception as e:
            logging.error(f"Error en get_visualization_data: {e}")
            return {"data": None, "type": viz_type, "error": str(e)}