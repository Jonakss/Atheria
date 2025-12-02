# src/batch_inference_engine.py
"""
Motor de inferencia batch para ejecutar múltiples simulaciones en paralelo.

Este es un prototipo para la Fase 1 de la arquitectura de inferencia masiva.
Permite ejecutar N simulaciones simultáneamente usando batching de PyTorch.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Optional, Dict, Any
import numpy as np

from .qca_engine import QuantumState, CartesianEngine
from .. import config as global_cfg


class BatchInferenceEngine:
    """
    Ejecuta múltiples simulaciones cuánticas en batch usando PyTorch.
    
    Ventajas sobre inferencia secuencial:
    - Mejor aprovechamiento de GPU (paralelización)
    - Throughput más alto (más simulaciones por segundo)
    - Eficiencia de memoria (comparte buffers)
    
    Ejemplo de uso:
        engine = BatchInferenceEngine(model, batch_size=32, device='cuda')
        engine.initialize_states(num_simulations=100)
        for step in range(1000):
            engine.evolve_batch(steps=1)
            if step % 100 == 0:
                stats = engine.get_batch_statistics()
    """
    
    def __init__(
        self,
        model: nn.Module,
        grid_size: int,
        d_state: int,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        cfg: Optional[Any] = None
    ):
        """
        Inicializa el motor de inferencia batch.
        
        Args:
            model: Modelo PyTorch (UNet, ConvLSTM, etc.)
            grid_size: Tamaño del grid (grid_size x grid_size)
            d_state: Dimensión del estado cuántico
            device: Dispositivo PyTorch (default: auto-detect)
            batch_size: Tamaño de batch para inferencia
            cfg: Configuración del experimento (para GAMMA_DECAY, etc.)
        """
        self.model = model.to(device or global_cfg.DEVICE)
        self.model.eval()  # Modo evaluación para inferencia
        self.grid_size = grid_size
        self.d_state = d_state
        self.device = device or global_cfg.DEVICE
        self.batch_size = batch_size
        self.cfg = cfg
        
        # Detectar si el modelo usa ConvLSTM
        self.has_memory = hasattr(model, 'convlstm') or 'ConvLSTM' in model.__class__.__name__
        
        # Estados de las simulaciones
        # Cada elemento es un QuantumState
        self.states: List[QuantumState] = []
        
        # Estados de memoria para ConvLSTM (si aplica)
        # Forma: [num_simulations, batch, channels, H, W]
        self.h_states: Optional[torch.Tensor] = None
        self.c_states: Optional[torch.Tensor] = None
        
        # Contador de pasos
        self.current_step = 0
        
        logging.info(
            f"BatchInferenceEngine inicializado: "
            f"grid_size={grid_size}, d_state={d_state}, "
            f"batch_size={batch_size}, device={self.device}, "
            f"has_memory={self.has_memory}"
        )
    
    def initialize_states(
        self,
        num_simulations: int,
        initial_mode: str = 'complex_noise',
        seed: Optional[int] = None
    ):
        """
        Inicializa N estados cuánticos para simulaciones paralelas.
        
        Args:
            num_simulations: Número de simulaciones a inicializar
            initial_mode: Modo de inicialización ('complex_noise', 'random', 'zeros')
            seed: Semilla aleatoria (opcional, para reproducibilidad)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.states = []
        for i in range(num_simulations):
            state = QuantumState(
                self.grid_size,
                self.d_state,
                self.device,
                initial_mode=initial_mode
            )
            self.states.append(state)
        
        # Inicializar memoria para ConvLSTM si es necesario
        if self.has_memory:
            # Obtener forma de memoria del modelo
            # Asumimos que el modelo tiene un método para obtener la forma
            # Por ahora, usamos una forma por defecto
            memory_shape = (1, 32, self.grid_size // 4, self.grid_size // 4)  # Ejemplo
            self.h_states = torch.zeros(
                num_simulations, *memory_shape,
                device=self.device,
                dtype=torch.float32
            )
            self.c_states = torch.zeros(
                num_simulations, *memory_shape,
                device=self.device,
                dtype=torch.float32
            )
        
        self.current_step = 0
        logging.info(f"Inicializados {num_simulations} estados cuánticos")
    
    def initialize_from_ionq_multiverse(self, num_universes: int, strength: float = 0.1):
        """
        Quantum Multiverse Initialization:
        Executes a SINGLE quantum circuit on IonQ with `shots=num_universes`.
        Each shot (measurement outcome) seeds a separate universe in the batch.
        
        This creates a "Many Worlds" simulation where each instance represents
        a different collapse of the same initial quantum wavefunction.
        """
        import logging
        from .compute_backend import IonQBackend
        from .. import config as cfg
        from qiskit import QuantumCircuit
        
        logging.info(f"🌌 Quantum Multiverse: Spawning {num_universes} universes from IonQ...")
        
        try:
            # 1. Setup Backend & Circuit
            backend = IonQBackend(api_key=cfg.IONQ_API_KEY, backend_name=cfg.IONQ_BACKEND_NAME)
            n_qubits = 11
            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits)) # Superposition
            for i in range(n_qubits - 1): # Entanglement
                qc.cx(i, i+1)
            qc.cx(n_qubits-1, 0)
            qc.measure_all()
            
            # 2. Execute ONE job with shots = num_universes
            # This is cost-efficient: 1 job credit for N universes
            counts = backend.execute('run_circuit', qc, shots=num_universes)
            
            # 3. Process results into individual seeds
            # We need exactly num_universes seeds. 
            # 'counts' gives us unique bitstrings and their frequency.
            # We expand this back into a list of bitstrings.
            bitstrings = []
            for bitstring, count in counts.items():
                bitstrings.extend([bitstring] * count)
            
            # Ensure we have enough (IonQ might return fewer shots if some failed, though rare)
            # Or if shots > num_universes requested (unlikely via API but possible logic wise)
            if len(bitstrings) < num_universes:
                # Pad with random choice if missing
                import random
                bitstrings.extend(random.choices(bitstrings, k=num_universes - len(bitstrings)))
            
            bitstrings = bitstrings[:num_universes]
            
            # 4. Initialize States
            self.states = []
            for i, bitstring in enumerate(bitstrings):
                # Convert bitstring to tensor noise
                # '0' -> -1, '1' -> 1
                bits = [1.0 if c == '1' else -1.0 for c in bitstring]
                quantum_data = torch.tensor(bits, device=self.device, dtype=torch.float32)
                
                # Tile to fill grid
                total_needed = self.grid_size * self.grid_size * self.d_state
                repeats = (total_needed // len(quantum_data)) + 1
                quantum_data = quantum_data.repeat(repeats)[:total_needed]
                noise = quantum_data.reshape(1, self.grid_size, self.grid_size, self.d_state)
                
                # Map to complex state
                noise = noise * strength
                real, imag = torch.cos(noise), torch.sin(noise)
                psi_universe = torch.complex(real, imag)
                
                # Create QuantumState with this specific seed
                state = QuantumState(
                    self.grid_size,
                    self.d_state,
                    self.device,
                    precomputed_state=psi_universe
                )
                self.states.append(state)
                
            logging.info(f"✨ Multiverse Created: {len(self.states)} universes initialized.")
            
            # Initialize memory if needed
            if hasattr(self.model, 'convlstm') or 'ConvLSTM' in self.model.__class__.__name__:
                memory_shape = (len(self.states), 64, self.grid_size, self.grid_size) # 64 is d_model usually
                self.h_states = torch.zeros(*memory_shape, device=self.device)
                self.c_states = torch.zeros(*memory_shape, device=self.device)
                
            self.current_step = 0
                
        except Exception as e:
            logging.error(f"❌ Quantum Multiverse Failed: {e}. Falling back to random.")
            self.initialize_states(num_universes, initial_mode='complex_noise')
    
    def evolve_batch(self, steps: int = 1):
        """
        Evoluciona todos los estados en batch por N pasos.
        
        Args:
            steps: Número de pasos a evolucionar
        """
        if not self.states:
            raise ValueError("No hay estados inicializados. Llama a initialize_states() primero.")
        
        num_simulations = len(self.states)
        
        with torch.no_grad():
            for step in range(steps):
                # Procesar en batches para no saturar memoria
                for batch_start in range(0, num_simulations, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, num_simulations)
                    batch_indices = range(batch_start, batch_end)
                    
                    # Obtener estados del batch actual
                    psi_batch = torch.stack([
                        self.states[i].psi for i in batch_indices
                    ])  # [batch_size, grid_size, grid_size, d_state]
                    
                    # Evolucionar batch
                    new_psi_batch = self._evolve_batch_logic(psi_batch, batch_indices)
                    
                    # Actualizar estados
                    for idx, i in enumerate(batch_indices):
                        self.states[i].psi = new_psi_batch[idx]
                
                self.current_step += 1
    
    def _evolve_batch_logic(
        self,
        psi_batch: torch.Tensor,
        batch_indices: range
    ) -> torch.Tensor:
        """
        Lógica de evolución para un batch de estados.
        
        Similar a CartesianEngine._evolve_logic pero para batch.
        
        Args:
            psi_batch: Tensor de estados [batch_size, H, W, d_state] (complejo)
            batch_indices: Índices del batch (para memoria ConvLSTM)
        
        Returns:
            Nuevos estados evolucionados [batch_size, H, W, d_state] (complejo)
        """
        # Preparar entrada para el modelo
        # Forma esperada: [batch, 2*d_state, H, W]
        x_cat_real = psi_batch.real.permute(0, 3, 1, 2)
        x_cat_imag = psi_batch.imag.permute(0, 3, 1, 2)
        x_cat_total = torch.cat([x_cat_real, x_cat_imag], dim=1)
        
        # Si el modelo tiene memoria (ConvLSTM)
        if self.has_memory and self.h_states is not None:
            # Obtener estados de memoria para este batch
            h_batch = self.h_states[batch_indices]  # [batch_size, ...]
            c_batch = self.c_states[batch_indices]  # [batch_size, ...]
            
            # Detach para evitar problemas con backward
            h_batch = h_batch.detach() if h_batch is not None else None
            c_batch = c_batch.detach() if c_batch is not None else None
            
            # Llamar al modelo con memoria
            delta_psi_unitario_complex, h_next, c_next = self.model(
                x_cat_total, h_batch, c_batch
            )
            
            # Actualizar memoria
            if h_next is not None:
                self.h_states[batch_indices] = h_next.detach()
            if c_next is not None:
                self.c_states[batch_indices] = c_next.detach()
        else:
            # Modelo sin memoria
            delta_psi_unitario_complex = self.model(x_cat_total)
        
        # Convertir salida a complejo
        delta_real, delta_imag = torch.chunk(delta_psi_unitario_complex, 2, dim=1)
        delta_psi_unitario = torch.complex(delta_real, delta_imag).permute(0, 2, 3, 1)
        
        # Aplicar término Lindbladian si está configurado
        gamma_decay = 0.0
        if self.cfg is not None:
            gamma_decay = getattr(self.cfg, 'GAMMA_DECAY', 0.0)
        
        if gamma_decay > 0:
            delta_psi_decay = -gamma_decay * psi_batch
            delta_psi_total = delta_psi_unitario + delta_psi_decay
        else:
            delta_psi_total = delta_psi_unitario
        
        # Método de Euler
        new_psi = psi_batch + delta_psi_total
        
        # Normalizar
        norm = torch.sqrt(torch.sum(new_psi.abs().pow(2), dim=-1, keepdim=True))
        new_psi = new_psi / (norm + 1e-9)
        
        return new_psi
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """
        Calcula estadísticas agregadas de todos los estados.
        
        Returns:
            Diccionario con estadísticas (energía promedio, entropía, etc.)
        """
        if not self.states:
            return {}
        
        energies = []
        entropies = []
        
        for state in self.states:
            psi = state.psi
            # Energía: |psi|²
            energy = torch.sum(psi.abs().pow(2)).item()
            energies.append(energy)
            
            # Entropía: -sum(p * log(p)) donde p = |psi|²
            prob = psi.abs().pow(2)
            prob = prob / (prob.sum() + 1e-9)
            entropy = -torch.sum(prob * torch.log(prob + 1e-9)).item()
            entropies.append(entropy)
        
        return {
            'num_simulations': len(self.states),
            'current_step': self.current_step,
            'avg_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'avg_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
        }
    
    def get_state(self, index: int) -> QuantumState:
        """
        Obtiene el estado de una simulación específica.
        
        Args:
            index: Índice de la simulación (0 a num_simulations-1)
        
        Returns:
            QuantumState de la simulación
        """
        if index < 0 or index >= len(self.states):
            raise IndexError(f"Índice {index} fuera de rango [0, {len(self.states)})")
        return self.states[index]
    
    def get_all_states(self) -> List[QuantumState]:
        """
        Obtiene todos los estados (útil para análisis).
        
        Returns:
            Lista de todos los QuantumState
        """
        return self.states
    
    def reset_state(self, index: int, initial_mode: str = 'complex_noise'):
        """
        Reinicia un estado específico.
        
        Args:
            index: Índice de la simulación a reiniciar
            initial_mode: Modo de inicialización
        """
        if index < 0 or index >= len(self.states):
            raise IndexError(f"Índice {index} fuera de rango")
        
        self.states[index] = QuantumState(
            self.grid_size,
            self.d_state,
            self.device,
            initial_mode=initial_mode
        )
    
    def clear(self):
        """Limpia todos los estados."""
        self.states = []
        self.h_states = None
        self.c_states = None
        self.current_step = 0
        logging.info("BatchInferenceEngine limpiado")


# Función de utilidad para crear un engine desde un experimento
def create_batch_engine_from_experiment(
    exp_name: str,
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> BatchInferenceEngine:
    """
    Crea un BatchInferenceEngine desde un experimento guardado.
    
    Args:
        exp_name: Nombre del experimento
        batch_size: Tamaño de batch
        device: Dispositivo (opcional)
    
    Returns:
        BatchInferenceEngine configurado
    """
    from .utils import load_experiment_config, get_latest_checkpoint
    from .model_loader import load_model
    
    # Cargar configuración
    config = load_experiment_config(exp_name)
    if not config:
        raise ValueError(f"Experimento '{exp_name}' no encontrado")
    
    # Cargar modelo
    checkpoint_path = get_latest_checkpoint(exp_name)
    if not checkpoint_path:
        raise ValueError(f"No hay checkpoint para '{exp_name}'")
    
    model, _ = load_model(config, checkpoint_path)
    if model is None:
        raise ValueError(f"Error al cargar modelo de '{exp_name}'")
    
    # Crear engine
    engine = BatchInferenceEngine(
        model=model,
        grid_size=global_cfg.GRID_SIZE_INFERENCE,
        d_state=config.MODEL_PARAMS.d_state,
        device=device,
        batch_size=batch_size,
        cfg=config
    )
    
    return engine

