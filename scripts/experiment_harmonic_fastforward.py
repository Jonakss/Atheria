import os
import sys
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Add project root to path (Insert at 0 to prioritize local src)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.engines.compute_backend import IonQBackend
from src.models.unet_unitary import UNetUnitary

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridPipeline:
    """
    Pipeline Híbrido:
    1. Quantum QFT (IonQ/Aer) -> Dominio de Frecuencia
    2. Classical UNet -> Evolución Temporal (Fast Forward)
    3. Quantum IQFT (IonQ/Aer) -> Dominio Espacial
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.api_key = os.getenv('IONQ_API_KEY')
        self.backend = None
        
        # Inicializar Backend Cuántico
        if self.api_key:
            try:
                self.backend = IonQBackend(api_key=self.api_key, backend_name=config.IONQ_BACKEND_NAME)
                logging.info(f"🔌 Connected to IonQ Backend: {config.IONQ_BACKEND_NAME}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to connect to IonQ: {e}")
        
        if not self.backend:
            logging.info("⚠️ Using Local Aer Simulator")
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()

        # Inicializar Modelo Clásico (UNet)
        # d_state=1 (real/imag as channels in QFT representation usually implies complex, 
        # but here we map 8 qubits -> 256 amplitudes. 
        # Let's assume we process the amplitude grid as an image.)
        # 8 Qubits = 2^8 = 256 states. If mapped to 16x16 grid.
        self.grid_size = 16
        self.n_qubits = 8
        self.model = UNetUnitary(d_state=1, hidden_channels=16).to(device)
        self.model.eval() # Mock mode for now (untrained)

    def quantum_qft(self, state_grid):
        """
        Aplica QFT a un estado 2D (16x16) usando un circuito cuántico.
        Retorna el espectro (amplitudes en base de frecuencia).
        """
        logging.info("⚛️ Executing Quantum QFT...")
        
        # 1. Codificación de Estado (Amplitude Encoding)
        # state_grid: [16, 16] tensor
        state_vector = state_grid.flatten().cpu().numpy()
        # Normalize
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
            
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(state_vector, range(self.n_qubits))
        
        # 2. QFT
        # Implementación manual o de librería. Qiskit tiene librería.
        from qiskit.circuit.library import QFT
        qc.append(QFT(self.n_qubits), range(self.n_qubits))
        # 3. Ejecución (Simulación de Estado Puro para UNet)
        # Usamos qiskit.quantum_info.Statevector para obtener el vector exacto
        # Esto es equivalente a un simulador de estado ideal
        try:
            sv = Statevector(qc)
            return torch.from_numpy(np.array(sv.data)).to(self.device)
            
        except Exception as e:
            logging.error(f"❌ QFT Failed: {e}")
            return None

    def neural_evolution(self, spectrum):
        """
        Aplica la evolución temporal usando la UNet en el dominio de la frecuencia.
        """
        logging.info("🧠 Executing Neural Fast Forward (UNet)...")
        
        # spectrum: [256] complex tensor
        # Reshape to [1, 2, 16, 16] for UNet (Real/Imag channels)
        # UNet expects [Batch, Channels, H, W]
        
        grid = spectrum.view(16, 16)
        real = grid.real
        imag = grid.imag
        
        # Stack as channels: [1, 2, 16, 16]
        input_tensor = torch.stack([real, imag], dim=0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            # UNet output: delta_psi or new_psi
            # UNetUnitary output channels = 2 * d_state = 2
            output = self.model(input_tensor)
            
        # Reconstruct complex tensor
        out_real = output[0, 0]
        out_imag = output[0, 1]
        new_spectrum = torch.complex(out_real, out_imag)
        
        return new_spectrum.flatten()

    def quantum_iqft(self, spectrum):
        """
        Aplica Inverse QFT para retornar al dominio espacial.
        """
        logging.info("⚛️ Executing Quantum IQFT...")
        
        # Convertir a numpy complex128 para precisión
        state_vector = spectrum.detach().cpu().numpy().astype(np.complex128)
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
            
        qc = QuantumCircuit(self.n_qubits)
        # Usar normalize=True si está disponible, pero mejor asegurar la norma manualmente
        qc.initialize(state_vector, range(self.n_qubits))
        
        from qiskit.circuit.library import QFT
        qc.append(QFT(self.n_qubits, inverse=True), range(self.n_qubits))
        qc.measure_all() # Ahora sí medimos para obtener la distribución final
        
        # Ejecución
        try:
            if isinstance(self.backend, IonQBackend):
                # Aquí sí podemos usar IonQ real/simulador para obtener counts
                # Transpilar
                # Nota: initialize usa 'reset', que IonQ no soporta.
                # Si falla, hacemos fallback a Aer.
                try:
                    qc_transpiled = transpile(qc, self.backend.backend)
                    counts = self.backend.execute('run_circuit', qc_transpiled, shots=1024)
                except Exception as ionq_error:
                    logging.warning(f"⚠️ IonQ execution failed ({ionq_error}). Falling back to Aer for IQFT.")
                    from qiskit_aer import AerSimulator
                    sim = AerSimulator()
                    qc_transpiled = transpile(qc, sim)
                    job = sim.run(qc_transpiled, shots=1024)
                    counts = job.result().get_counts()
            else:
                qc_transpiled = transpile(qc, self.backend)
                job = self.backend.run(qc_transpiled, shots=1024)
                counts = job.result().get_counts()
                
            return counts
        except Exception as e:
            logging.error(f"❌ IQFT Failed: {e}")
            return {}

def main():
    print("🚀 Iniciando Experimento Híbrido: Harmonic UNet Fast Forward")
    
    # 1. Generar Estado Inicial (Pulso Gaussiano)
    x = np.linspace(-2, 2, 16)
    y = np.linspace(-2, 2, 16)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    initial_state = np.exp(-R**2) # Gaussiana
    initial_tensor = torch.from_numpy(initial_state).float()
    
    print("\n1️⃣  Estado Inicial Generado (Gaussiana 16x16)")
    
    # 2. Inicializar Pipeline
    pipeline = HybridPipeline()
    
    # 3. Paso 1: Quantum QFT
    print("\n2️⃣  Ejecutando QFT (Quantum Fourier Transform)...")
    spectrum = pipeline.quantum_qft(initial_tensor)
    if spectrum is None:
        return
        
    print(f"   ✅ Espectro obtenido. Shape: {spectrum.shape}")
    
    # 4. Paso 2: Neural Fast Forward
    print("\n3️⃣  Ejecutando Neural Fast Forward (UNet)...")
    evolved_spectrum = pipeline.neural_evolution(spectrum)
    print("   ✅ Espectro evolucionado por IA.")
    
    # 5. Paso 3: Quantum IQFT
    print("\n4️⃣  Ejecutando IQFT (Inverse QFT) y Medición...")
    final_counts = pipeline.quantum_iqft(evolved_spectrum)
    
    print("\n📊 Resultados Finales (Top 10 estados):")
    sorted_counts = sorted(final_counts.items(), key=lambda item: item[1], reverse=True)
    for state, count in sorted_counts[:10]:
        print(f"   |{state}> : {count}")

if __name__ == "__main__":
    main()
