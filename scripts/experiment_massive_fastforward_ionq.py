import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, Diagonal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.engines.compute_backend import IonQBackend
from src.models.unet_unitary import UNetUnitary

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 Iniciando EXP-007-IonQ: Massive Fast Forward (Hardware/Job Mode)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Cargar Checkpoint y Extraer Operador (Igual que antes)
    checkpoint_path = "output/UNET_UNITARIA-D8-H32-G64-LR1e-4/output/checkpoints/UNetUnitary_G64_Eps130_1762992979_FINAL.pth"
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return

    logging.info(f"📂 Cargando modelo desde: {checkpoint_path}")
    d_state = 8
    hidden_channels = 32
    model = UNetUnitary(d_state=d_state, hidden_channels=hidden_channels).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        logging.error(f"❌ Error cargando pesos: {e}")
        return

    model.eval()

    # 2. Extraer Operador Efectivo
    logging.info("🔮 Extrayendo Operador Holográfico Efectivo...")
    
    # MAXIMIZANDO QUBITS (Dentro de límites de API payload)
    # IonQ Simulator soporta 29 qubits.
    # Pero una compuerta Diagonal requiere enviar 2^N parámetros complejos.
    # - 10 qubits (32x32) = 1k params (OK)
    # - 16 qubits (256x256) = 65k params (FAIL - TooManyGates en descomposición)
    # Bajamos a 32x32 (10 qubits) para asegurar ejecución exitosa.
    grid_size = 32 
    n_qubits = int(np.log2(grid_size * grid_size))
    logging.info(f"   Grid Size: {grid_size}x{grid_size} -> {n_qubits} Qubits")
    
    probe_input = torch.randn(1, 2 * d_state, grid_size, grid_size, device=device)
    probe_input = probe_input / torch.norm(probe_input)
    
    with torch.no_grad():
        # Interpolación dinámica para adaptar el modelo de 64x64 al grid de 256x256
        probe_input_64 = torch.nn.functional.interpolate(probe_input, size=(64, 64), mode='bilinear')
        delta_psi_64 = model(probe_input_64)
        probe_output_64 = probe_input_64 + delta_psi_64
        probe_output = torch.nn.functional.interpolate(probe_output_64, size=(grid_size, grid_size), mode='bilinear')
    
    input_complex = torch.complex(probe_input[:, 0], probe_input[:, 1])
    output_complex = torch.complex(probe_output[:, 0], probe_output[:, 1])
    
    epsilon = 1e-8
    input_freq = torch.fft.fft2(input_complex)
    output_freq = torch.fft.fft2(output_complex)
    transfer_function = output_freq / (input_freq + epsilon)
    
    # 3. Potenciación (Fast Forward)
    N_STEPS = 1_000_000
    logging.info(f"⏩ Calculando Fast Forward de {N_STEPS} pasos...")
    
    transfer_function_unitary = transfer_function / (transfer_function.abs() + epsilon)
    r = transfer_function_unitary.abs()
    theta = transfer_function_unitary.angle()
    theta_final = theta * N_STEPS
    transfer_function_final = torch.polar(torch.ones_like(r), theta_final)
    
    # Aplanar y casting
    diagonal_phases = transfer_function_final.flatten().detach().cpu().numpy().astype(np.complex128)
    
    # 4. Construir Circuito
    logging.info("⚛️ Construyendo Circuito Cuántico (16 Qubits)...")
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits)) # Estado inicial uniforme
    qc.append(QFT(n_qubits), range(n_qubits))
    qc.append(Diagonal(diagonal_phases), range(n_qubits))
    qc.append(QFT(n_qubits, inverse=True), range(n_qubits))
    qc.measure_all()
    
    # 5. Enviar a IonQ
    api_key = os.getenv('IONQ_API_KEY')
    if not api_key:
        logging.error("❌ No IONQ_API_KEY found.")
        return

    backend = IonQBackend(api_key=api_key, backend_name='ionq_simulator')
    logging.info(f"🔌 Enviando Job a IonQ ({backend.backend_name})...")
    
    qc_transpiled = transpile(qc, backend.backend)
    
    try:
        job = backend.backend.run(qc_transpiled, shots=1024)
        job_id = job.job_id()
        logging.info(f"🚀 Job enviado! ID: {job_id}")
        logging.info("⏳ Esperando resultados...")
        
        result = job.result()
        counts = result.get_counts()
        logging.info("✅ Resultados recibidos de IonQ.")
        
        # 6. CERRAR EL CICLO: Reconstrucción y Continuación
        logging.info("🔄 Cerrando el ciclo: Reconstruyendo estado para inferencia...")
        
        # Reconstruir imagen desde counts
        # Counts es un dict {'00...01': 5, ...}
        # Mapeamos a tensor 2D
        reconstructed_image = torch.zeros(grid_size * grid_size, device=device)
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Qiskit es Little Endian, pero nuestra codificación espacial suele ser Big Endian o natural.
            # Asumiremos orden natural (int(bitstring)) para este demo.
            idx = int(bitstring, 2)
            if idx < reconstructed_image.numel():
                reconstructed_image[idx] = count / total_shots
        
        reconstructed_image = reconstructed_image.view(1, 1, grid_size, grid_size)
        # Raíz cuadrada para obtener amplitud (probabilidad -> amplitud)
        reconstructed_image = torch.sqrt(reconstructed_image)
        
        # Normalizar
        reconstructed_image = reconstructed_image / (reconstructed_image.norm() + 1e-8)
        
        logging.info("🧠 Ejecutando Inferencia de Continuación (1 paso)...")
        # Adaptar al input del modelo (2*d_state canales, 64x64)
        # Reconstructed image es [1, 1, H, W] (Amplitud)
        # Asumimos que esta amplitud va al primer canal (Real) y el resto es 0 o ruido.
        # d_state = 8 -> 16 canales totales.
        
        # Crear tensor base de ceros [1, 16, H, W]
        cont_input_full = torch.zeros(1, 2 * d_state, grid_size, grid_size, device=device)
        # Asignar la imagen reconstruida al primer canal
        cont_input_full[:, 0:1, :, :] = reconstructed_image
        
        # Interpolamos todo el tensor de 16 canales a 64x64
        cont_input_64 = torch.nn.functional.interpolate(cont_input_full, size=(64, 64), mode='bilinear')
        
        with torch.no_grad():
            delta_cont = model(cont_input_64)
            cont_output = cont_input_64 + delta_cont
            
        logging.info("✅ Ciclo Completado: IonQ Output -> PyTorch Tensor -> UNet Inference")
        
        # Guardar todo
        torch.save({
            'job_id': job_id,
            'counts': counts,
            'reconstructed_state': reconstructed_image,
            'continued_state': cont_output,
            'description': 'IonQ Massive Fast Forward + Continuity Loop'
        }, "output/checkpoints/fastforward_1M_ionq_loop.pt")
        logging.info("💾 Checkpoint de ciclo guardado.")
        
    except Exception as e:
        logging.error(f"❌ Error en ejecución IonQ: {e}")

if __name__ == "__main__":
    main()
