import torch
import logging
import numpy as np
from .. import config as cfg

class QuantumSteering:
    """
    Módulo de 'Quantum Steering' (Quantum Brush).
    
    Permite al usuario inyectar patrones cuánticos específicos (Vórtices, Solitones, etc.)
    en la simulación en tiempo real, usando IonQ para generar la estructura de fase.
    """
    def __init__(self, device):
        self.device = device
        self.backend = None
        self._setup_backend()
        
    def _setup_backend(self):
        try:
            from ..engines.compute_backend import IonQBackend
            if cfg.IONQ_API_KEY:
                self.backend = IonQBackend(api_key=cfg.IONQ_API_KEY, backend_name=cfg.IONQ_BACKEND_NAME)
            else:
                logging.warning("⚠️ IonQ API Key missing. QuantumSteering will run in MOCK mode.")
        except Exception as e:
            logging.error(f"❌ Failed to init IonQBackend for Steering: {e}")

    def generate_pattern(self, pattern_type, size=11):
        """
        Genera un patrón cuántico usando un circuito específico en IonQ.
        
        Args:
            pattern_type: 'vortex', 'soliton', 'superposition', 'entanglement'
            size: Tamaño del patrón (n_qubits). Default 11 (IonQ basic).
            
        Returns:
            Tensor complejo 1D de tamaño [size]
        """
        if self.backend is None:
            return self._mock_pattern(pattern_type, size)
            
        try:
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(size)
            
            if pattern_type == 'vortex':
                # Vórtice: Fase rotacional.
                # Usamos puertas de fase progresiva Rz
                qc.h(range(size)) # Superposición base
                for i in range(size):
                    angle = 2 * np.pi * (i / size) # 0 a 2pi
                    qc.rz(angle, i)
                # Entrelazamiento para coherencia
                for i in range(size-1):
                    qc.cx(i, i+1)
                    
            elif pattern_type == 'soliton':
                # Solitón: Paquete de ondas localizado.
                # Alta probabilidad en el centro, fase coherente.
                mid = size // 2
                qc.x(mid) # Excitar centro
                # Difusión controlada
                if mid > 0: qc.cx(mid, mid-1)
                if mid < size-1: qc.cx(mid, mid+1)
                qc.h(range(size)) # Fase
                
            elif pattern_type == 'entanglement':
                # Bell Pairs distribuidos
                for i in range(0, size-1, 2):
                    qc.h(i)
                    qc.cx(i, i+1)
                    
            else: # Superposition (default)
                qc.h(range(size))
            
            qc.measure_all()
            
            # Ejecutar
            counts = self.backend.execute('run_circuit', qc, shots=128)
            
            # Reconstruir estado promedio
            # Promediamos los bitstrings para obtener "amplitud"
            # Y usamos una fase sintética basada en el tipo si no podemos medirla directamente
            # (En QC real, medir destruye fase, pero aquí simulamos el efecto "generativo")
            
            amplitudes = np.zeros(size)
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                bits = np.array([1 if c == '1' else 0 for c in bitstring])
                # Ajustar longitud si mismatch
                if len(bits) > size: bits = bits[:size]
                elif len(bits) < size: bits = np.pad(bits, (0, size-len(bits)))
                
                amplitudes += bits * count
                
            amplitudes = amplitudes / total_shots
            amplitudes = torch.tensor(amplitudes, device=self.device, dtype=torch.float32)
            
            # Generar fase según patrón (ya que la medición la perdió)
            if pattern_type == 'vortex':
                phases = torch.tensor([2 * np.pi * (i / size) for i in range(size)], device=self.device)
            else:
                phases = torch.zeros(size, device=self.device)
                
            return torch.complex(amplitudes * torch.cos(phases), amplitudes * torch.sin(phases))
            
        except Exception as e:
            logging.error(f"❌ Steering Generation Failed: {e}")
            return self._mock_pattern(pattern_type, size)

    def _mock_pattern(self, pattern_type, size):
        """Generador local de patrones."""
        x = torch.linspace(-1, 1, size, device=self.device)
        
        if pattern_type == 'vortex':
            # Fase rotando
            mag = torch.exp(-x**2 * 2) # Gaussiana
            phase = x * np.pi
            return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
            
        elif pattern_type == 'soliton':
            # Pico estrecho
            mag = torch.exp(-x**2 * 10)
            return torch.complex(mag, torch.zeros_like(mag))
            
        else:
            return torch.randn(size, dtype=torch.complex64, device=self.device)

    def inject(self, state, pattern_type, x, y, radius=5):
        """
        Inyecta el patrón en el estado en la posición (x, y).
        """
        H, W = state.shape[1], state.shape[2]
        
        # Generar patrón 1D
        pattern_1d = self.generate_pattern(pattern_type, size=radius*2)
        
        # Crear patrón 2D (producto exterior aproximado o rotación)
        # Para simplificar: producto exterior de patrón consigo mismo
        pattern_2d = torch.outer(pattern_1d, pattern_1d).unsqueeze(0).unsqueeze(-1) # [1, 2r, 2r, 1]
        pattern_2d = pattern_2d.repeat(1, 1, 1, state.shape[-1]) # Repetir canales
        
        # Coordenadas
        y_start = max(0, y - radius)
        y_end = min(H, y + radius)
        x_start = max(0, x - radius)
        x_end = min(W, x + radius)
        
        # Recortar patrón si se sale del borde
        pat_y_start = 0 if y - radius >= 0 else -(y - radius)
        pat_y_end = pat_y_start + (y_end - y_start)
        pat_x_start = 0 if x - radius >= 0 else -(x - radius)
        pat_x_end = pat_x_start + (x_end - x_start)
        
        # Inyectar (suma o reemplazo? Suma conserva info previa, reemplazo es "pincel")
        # Vamos con mezcla suave
        alpha = 0.8
        
        current_slice = state[:, y_start:y_end, x_start:x_end, :]
        pattern_slice = pattern_2d[:, pat_y_start:pat_y_end, pat_x_start:pat_x_end, :]
        
        # Asegurar dimensiones coinciden
        if current_slice.shape != pattern_slice.shape:
            return state # Fallback seguro
            
        new_slice = current_slice * (1 - alpha) + pattern_slice * alpha
        
        new_state = state.clone()
        new_state[:, y_start:y_end, x_start:x_end, :] = new_slice
        
        logging.info(f"🖌️ Quantum Brush: Injected '{pattern_type}' at ({x}, {y})")
        return new_state
