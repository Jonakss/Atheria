import torch
import logging
import numpy as np
from .. import config as cfg

class IonQCollapse:
    """
    Módulo de Colapso Cuántico Híbrido.
    
    Permite que el estado de la simulación (clásico/neural) interactúe con un
    procesador cuántico real (IonQ).
    
    Flujo:
    1. Encoding: El estado local se codifica en parámetros de un circuito (ángulos).
    2. Procesamiento: IonQ evoluciona y colapsa (mide) el estado.
    3. Decoding: El resultado de la medición actualiza el estado de la simulación.
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
                logging.info(f"⚛️ IonQCollapse initialized with backend: {cfg.IONQ_BACKEND_NAME}")
            else:
                logging.warning("⚠️ IonQ API Key missing. IonQCollapse will run in MOCK mode.")
        except Exception as e:
            logging.error(f"❌ Failed to init IonQBackend: {e}")

    def collapse(self, state_tensor, region_center=None, intensity=0.1):
        """
        Realiza un colapso de función de onda en una región específica.
        
        Args:
            state_tensor: Tensor de estado completo [1, H, W, d_state] (complejo)
            region_center: Tupla (y, x) del centro del colapso. Si es None, aleatorio.
            intensity: Fuerza del efecto (mezcla entre estado original y colapsado).
            
        Returns:
            Nuevo estado con la región actualizada.
        """
        if self.backend is None:
            return self._mock_collapse(state_tensor, intensity)
            
        try:
            H, W = state_tensor.shape[1], state_tensor.shape[2]
            
            # 1. Seleccionar Región (11 qubits max para IonQ basic)
            # Usaremos una línea o bloque pequeño. Digamos 1x11 o 3x3 (9 qubits).
            # Vamos con 11 qubits lineales para simplificar mapeo directo.
            n_qubits = 11
            
            if region_center is None:
                cy, cx = np.random.randint(0, H), np.random.randint(0, W)
            else:
                cy, cx = region_center
                
            # Extraer datos locales para encoding
            # Tomamos 11 células horizontales centradas en cx
            start_x = max(0, cx - n_qubits // 2)
            end_x = min(W, start_x + n_qubits)
            # Ajustar start si end se salió
            if end_x - start_x < n_qubits:
                start_x = max(0, end_x - n_qubits)
            
            # Extraer magnitud/fase para codificar
            # Shape: [1, 1, n_qubits, d_state] -> tomamos canal 0 por simplicidad o promedio
            region_slice = state_tensor[0, cy, start_x:end_x, :]
            
            # Encoding: Fase media del estado -> Ángulo de rotación Ry
            # region_slice es complex. Angle = fase.
            phases = torch.angle(region_slice).mean(dim=-1).cpu().numpy() # [n_qubits]
            
            # Si faltan qubits (bordes), rellenar con 0
            if len(phases) < n_qubits:
                phases = np.pad(phases, (0, n_qubits - len(phases)))
                
            # 2. Construir Circuito Variacional (State-Dependent)
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(n_qubits)
            
            # Encoding Layer
            for i in range(n_qubits):
                # Mapeamos la fase [-pi, pi] a rotación Ry
                qc.ry(phases[i], i)
                
            # Interaction Layer (Entanglement)
            for i in range(n_qubits - 1):
                qc.cx(i, i+1)
            qc.measure_all()
            
            # 3. Ejecutar en IonQ
            # 1 shot es suficiente para un "colapso" único
            counts = self.backend.execute('run_circuit', qc, shots=1)
            
            # 4. Decoding
            measured_bitstring = list(counts.keys())[0] # Ej: "10110..."
            
            # Convertir bitstring a valores físicos (+1/-1)
            collapse_vals = torch.tensor(
                [1.0 if c == '1' else -1.0 for c in measured_bitstring],
                device=self.device, dtype=torch.float32
            )
            
            # Actualizar estado
            # Mezclamos el estado colapsado con el original
            # El colapso afecta la MAGNITUD (partícula detectada o no) y resetea FASE
            
            # Crear máscara de actualización
            update_mask = torch.zeros_like(state_tensor.real)
            
            # Llenar la región afectada
            # Expandimos collapse_vals a (n_qubits, d_state)
            collapse_expanded = collapse_vals.unsqueeze(1).repeat(1, state_tensor.shape[-1])
            
            # Aplicar
            current_region = state_tensor[0, cy, start_x:end_x, :]
            
            # Nueva magnitud: forzada por el colapso (0 o 1, suavizado)
            # Si bit=1 -> Magnitud alta. Si bit=0 -> Magnitud baja.
            # Mapeo: -1 -> 0.1, 1 -> 1.0
            target_mag = (collapse_vals + 1) / 2.0 
            target_mag = target_mag * 0.9 + 0.1 # [0.1, 1.0]
            
            # Nueva fase: Aleatoria o 0 (reset post-medición)
            # En QM, la fase se vuelve indefinida o se proyecta. Dejémosla en 0.
            target_phase = torch.zeros_like(target_mag)
            
            target_complex = torch.complex(
                target_mag * torch.cos(target_phase),
                target_mag * torch.sin(target_phase)
            ).unsqueeze(1).repeat(1, state_tensor.shape[-1])

            # Interpolación (Intensity)
            # new = old * (1-intensity) + target * intensity
            # Pero solo en la región

            # Necesitamos rebanar el tensor original para escribir
            # PyTorch no soporta asignación compleja directa en slices a veces, cuidado

            new_region = current_region * (1 - intensity) + target_complex[:len(current_region)] * intensity

            # Escribir de vuelta (clonamos para no mutar in-place si afecta autograd)
            new_state = state_tensor.clone()
            new_state[0, cy, start_x:end_x, :] = new_region

            logging.info(f"⚡ IonQ Collapse at ({cx}, {cy}): {measured_bitstring}")
            return new_state

        except Exception as e:
            logging.error(f"❌ IonQ Collapse Failed: {e}")
            return self._mock_collapse(state_tensor, intensity)

    def _mock_collapse(self, state_tensor, intensity):
        """Simulación local de colapso (ruido)."""
        noise = torch.randn_like(state_tensor) * intensity
        return state_tensor + noise
