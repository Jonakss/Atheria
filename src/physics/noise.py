import torch

class QuantumNoiseInjector:
    """
    Simula el entorno hostil que fuerza la evolución de la simetría y la robustez.
    Basado en principios de corrección de errores cuánticos (IonQ / Bacon-Shor).
    """
    def __init__(self, device):
        self.device = device

    def apply_phase_flip(self, psi, rate=0.01):
        """
        Ruido Cuántico (Z-Error).
        Rota aleatoriamente la fase sin cambiar la magnitud (energía).
        
        Efecto Evolutivo: Destruye la coherencia necesaria para el movimiento.
        Solución que busca la IA: Redundancia de fase y simetría paridad.
        """
        # Máscara booleana de dónde ocurre el error
        mask = torch.rand(psi.shape[:3], device=self.device) < rate
        mask = mask.unsqueeze(-1).expand_as(psi)
        
        # Rotación aleatoria (0 a 2pi)
        random_angle = torch.rand(psi.shape, device=self.device) * 2 * torch.pi
        phase_noise = torch.complex(torch.cos(random_angle), torch.sin(random_angle))
        
        psi_out = psi.clone()
        psi_out[mask] = psi[mask] * phase_noise[mask]
        return psi_out

    def apply_bit_flip(self, psi, rate=0.005):
        """
        Ruido Clásico (X-Error).
        Invierte la amplitud o satura la energía. Simula 'rayos cósmicos'.
        
        Efecto Evolutivo: Destruye la estructura local.
        Solución que busca la IA: Robustez estructural (partículas densas).
        """
        mask = torch.rand(psi.shape[:3], device=self.device) < rate
        mask = mask.unsqueeze(-1).expand_as(psi)
        
        psi_out = psi.clone()
        # Inversión de signo drástica (Flip de 180 grados)
        psi_out[mask] = -psi[mask]
        return psi_out
    
    def apply_thermal_jitter(self, psi, temperature=0.01):
        """
        Ruido Térmico (Aditivo).
        Pequeñas fluctuaciones constantes en todo el campo.
        Evita que el sistema se congele en mínimos locales.
        """
        noise_real = torch.randn_like(psi.real) * temperature
        noise_imag = torch.randn_like(psi.imag) * temperature
        noise = torch.complex(noise_real, noise_imag)
        return psi + noise

    def apply_ionq_noise(self, psi, rate=0.01):
        """
        Ruido Cuántico Real (IonQ).
        Usa mediciones de un QPU para generar máscaras de error.
        
        Estrategia de Buffer:
        Para evitar latencia y consumo excesivo de créditos, pedimos un batch grande
        de entropía cuántica y lo consumimos poco a poco.
        """
        if not hasattr(self, 'noise_buffer') or len(self.noise_buffer) < psi.numel() * 0.1:
            self._refill_noise_buffer(psi.numel())
            
        # Consumir del buffer
        needed = int(psi.numel() * rate) # Solo necesitamos bits para los errores
        # Simplificación: Usamos el buffer para generar la máscara
        # Tomamos un chunk del buffer y lo usamos como semilla o directamente como ruido
        
        # Estrategia eficiente:
        # Usar el buffer para sembrar un generador local (Hybrid Randomness)
        # O usar el buffer directamente para la máscara (True Randomness)
        
        # Vamos con True Randomness para la máscara
        # Necesitamos un tensor del tamaño de psi
        # Si el buffer es pequeño, lo repetimos (tiling)
        
        buffer_tensor = self.noise_buffer
        if buffer_tensor.numel() < psi.numel():
             repeats = (psi.numel() // buffer_tensor.numel()) + 1
             buffer_tensor = buffer_tensor.repeat(repeats)[:psi.numel()]
        else:
             buffer_tensor = buffer_tensor[:psi.numel()]
             
        # Reshape
        noise_mask = buffer_tensor.reshape(psi.shape).to(self.device)
        
        # Rotar el buffer (circular buffer simple)
        self.noise_buffer = torch.roll(self.noise_buffer, shifts=-int(psi.numel()/10))
        
        # Aplicar ruido donde la máscara > umbral (ajustado por rate)
        # El buffer tiene valores -1 y 1.
        # Esto es un poco determinista si solo rotamos.
        # Mejor: Usar el buffer como fuente de entropía para decidir DÓNDE aplicar el error.
        
        # Enfoque Híbrido:
        # Usamos el buffer cuántico para modular la fase
        phase_noise = torch.complex(torch.cos(noise_mask * torch.pi), torch.sin(noise_mask * torch.pi))
        
        # Aplicar solo con probabilidad 'rate' (usando rand clásico para la selección escasa, 
        # pero el VALOR del ruido viene de IonQ)
        mask = torch.rand(psi.shape, device=self.device) < rate
        
        psi_out = psi.clone()
        psi_out[mask] = psi[mask] * phase_noise[mask]
        return psi_out

    def _refill_noise_buffer(self, min_size=10000):
        """Recarga el buffer de ruido desde IonQ."""
        try:
            from ..engines.compute_backend import IonQBackend
            from .. import config as cfg
            from qiskit import QuantumCircuit
            import logging
            
            logging.info("⚛️ Refilling Quantum Noise Buffer from IonQ...")
            
            backend = IonQBackend(api_key=cfg.IONQ_API_KEY, backend_name=cfg.IONQ_BACKEND_NAME)
            n_qubits = 11
            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits)) # Max Entropy
            qc.measure_all()
            
            # Pedir suficientes shots
            shots = 1024 # ~11k bits
            counts = backend.execute('run_circuit', qc, shots=shots)
            
            bit_stream = []
            for bitstring, count in counts.items():
                bits = [1.0 if c == '1' else -1.0 for c in bitstring]
                bit_stream.extend(bits * count)
            
            self.noise_buffer = torch.tensor(bit_stream, device=self.device, dtype=torch.float32)
            logging.info(f"✅ Noise Buffer Refilled: {self.noise_buffer.numel()} quantum bits.")
            
        except Exception as e:
            import logging
            logging.error(f"❌ Failed to refill noise buffer: {e}. Using pseudo-random fallback.")
            self.noise_buffer = torch.randn(min_size, device=self.device).sign()
