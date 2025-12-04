import os
import sys
import torch
import logging
from qiskit_ionq import IonQProvider
from qiskit import transpile

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from scripts.experiment_advanced_ansatz import StronglyEntanglingConv2d

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("🚀 Iniciando Verificación de Inferencia en IonQ...")
    
    # 1. Cargar Modelo Entrenado
    model_path = "output/models/quantum_fastforward_final.pt"
    if not os.path.exists(model_path):
        logging.error(f"❌ Modelo no encontrado: {model_path}")
        return

    logging.info(f"📂 Cargando modelo: {model_path}")
    cp = torch.load(model_path, map_location='cpu')
    
    # Reconstruir modelo
    config_dict = cp['config']
    grid_size = config_dict['grid_size']
    n_layers = config_dict['n_layers']
    
    logging.info(f"   Config: Grid {grid_size}x{grid_size}, {n_layers} Layers, Ansatz: {config_dict['ansatz']}")
    
    model = StronglyEntanglingConv2d(grid_size=grid_size, n_layers=n_layers, device='cpu')
    model.load_state_dict(cp['model_state_dict'])
    
    # 2. Generar Circuito
    logging.info("⚛️ Generando circuito cuántico...")
    qc = model.export_circuit()
    qc.measure_all() # Agregar mediciones para obtener resultados
    
    logging.info(f"   Profundidad: {qc.depth()}")
    logging.info(f"   Qubits: {qc.num_qubits}")
    
    # 3. Conectar a IonQ
    api_key = os.getenv('IONQ_API_KEY')
    if not api_key:
        logging.error("❌ IONQ_API_KEY no encontrada en variables de entorno.")
        return
        
    provider = IonQProvider(token=api_key)
    # Usar simulador para verificación rápida, o 'ionq_qpu' para real
    backend_name = os.getenv('IONQ_BACKEND_NAME', 'ionq_simulator') 
    backend = provider.get_backend(backend_name)
    
    logging.info(f"🔌 Conectado a backend: {backend.name}")
    
    # 4. Enviar Job
    logging.info("📤 Enviando trabajo a IonQ...")
    
    # Transpile for IonQ native gates
    qc_transpiled = transpile(qc, backend)
    
    job = backend.run(qc_transpiled, shots=1024)
    job_id = job.job_id()
    
    logging.info(f"✅ Job enviado exitosamente!")
    logging.info(f"🆔 JOB ID: {job_id}")
    
    with open("ionq_job_id.txt", "w") as f:
        f.write(job_id)
        
    logging.info(f"   Puedes ver el estado en https://cloud.ionq.com/")
    
    # 5. Esperar Resultados (Opcional, bloqueante)
    logging.info("⏳ Esperando resultados...")
    result = job.result()
    counts = result.get_counts()
    
    logging.info(f"📊 Resultados recibidos (Top 5):")
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    for state, count in sorted_counts:
        logging.info(f"   |{state}>: {count}")

if __name__ == "__main__":
    main()
