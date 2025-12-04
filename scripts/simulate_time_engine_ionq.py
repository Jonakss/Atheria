import os
import sys
import logging
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RZZGate, RXGate

# Añadir raíz del proyecto al path (ajusta según tu estructura de carpetas)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Intentar importar el backend de Atheria, o usar uno genérico si falla
try:
    from src.engines.compute_backend import IonQBackend
    from src import config
except ImportError:
    # Fallback para ejecución standalone
    class Config: IONQ_API_KEY = os.getenv("IONQ_API_KEY"); IONQ_BACKEND_NAME = "ionq_simulator"
    config = Config()
    IonQBackend = None 
    print("⚠️ Ejecutando en modo standalone (sin dependencias completas de Atheria)")

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [TIME_ENGINE] - %(message)s')

def build_time_engine_circuit(num_qubits, time_steps, dt, interaction_strength=1.0, field_strength=0.5):
    """
    Construye un circuito que actúa como un 'Motor de Tiempo'.
    Usa Trotterización para simular la evolución Hamiltoniana H = Σ J ZiZj + Σ h Xi
    
    Args:
        num_qubits: Número de células/qubits en el universo 1D.
        time_steps: Cuántos pasos de tiempo 'adelantar'.
        dt: Delta de tiempo por paso.
    """
    qc = QuantumCircuit(num_qubits)
    
    # 1. Estado Inicial (Genesis)
    # Ponemos los qubits en superposición para tener algo interesante que evolucionar
    qc.h(range(num_qubits))
    
    logging.info(f"⏳ Construyendo Motor de Tiempo: {num_qubits} qubits, {time_steps} pasos (dt={dt})")
    
    # 2. Bucle de Evolución Temporal (El "Engine")
    for step in range(time_steps):
        # Capa de Interacción (Vecinos: Zi Zj)
        # Esto simula la propagación de información entre células
        for i in range(num_qubits - 1):
            theta_zz = -2 * interaction_strength * dt
            qc.rzz(theta_zz, i, i+1)
            
        # Condición de frontera periódica (Cierra el anillo)
        qc.rzz(theta_zz, num_qubits-1, 0)
        
        # Capa de Campo Transversal (Auto-evolución: Xi)
        # Esto simula la dinámica interna de cada célula
        for i in range(num_qubits):
            theta_x = -2 * field_strength * dt
            qc.rx(theta_x, i)
            
        # Barrera visual para separar pasos de tiempo
        qc.barrier()

    # 3. Observación (Medición)
    qc.measure_all()
    
    return qc

def main():
    print("\n⚛️ INICIANDO SIMULACIÓN DE MOTOR DE TIEMPO IONQ ⚛️\n")
    
    # Configuración
    API_KEY = config.IONQ_API_KEY
    # if not API_KEY:
    #     print("❌ Error: No se encontró IONQ_API_KEY. Ejecuta 'export IONQ_API_KEY=...'")
    #     return

    # Parámetros del Universo
    NUM_QUBITS = 6   # Pequeño universo 1D
    STEPS = 3        # Pasos de tiempo a adelantar
    DT = 0.5         # Tamaño del salto temporal
    
    # Construir el circuito
    qc = build_time_engine_circuit(NUM_QUBITS, STEPS, DT)
    print(f"Planos del Motor Temporal ({STEPS} pasos):\n")
    print(qc.draw(output='text', idle_wires=False))
    
    # Inicializar Backend IonQ
    # Inicializar Backend IonQ
    use_aer = True
    if API_KEY:
        try:
            if IonQBackend:
                # Usar la clase wrapper de Atheria si está disponible
                backend = IonQBackend(api_key=API_KEY, backend_name=config.IONQ_BACKEND_NAME)
                print(f"\n🔌 Conectado a IonQ Backend: {config.IONQ_BACKEND_NAME}")
                print("🚀 Enviando circuito al futuro...")
                counts = backend.execute('run_circuit', qc, shots=1024)
                use_aer = False
            else:
                # Fallback directo a qiskit-ionq si no estamos en el entorno Atheria
                from qiskit_ionq import IonQProvider
                provider = IonQProvider(API_KEY)
                backend = provider.get_backend("ionq_simulator")
                print(f"\n🔌 Conectado a IonQ Provider directo")
                job = backend.run(qc, shots=1024)
                print("🚀 Enviando circuito al futuro...")
                counts = job.result().get_counts()
                use_aer = False
        except Exception as e:
            print(f"⚠️ Error conectando con IonQ: {e}")
            print("🔄 Cayendo a simulador local...")
            use_aer = True
    
    if use_aer:
        print("\n⚠️ Usando simulador local (Aer).")
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        job = backend.run(transpile(qc, backend), shots=1024)
        counts = job.result().get_counts()

    print("\n📊 Estado del Universo tras T={}:".format(STEPS * DT))
    print(counts)
    
    # Decodificar el estado más probable
    most_likely = max(counts, key=counts.get)
    print(f"\n🔮 Línea temporal dominante: |{most_likely}>")

if __name__ == "__main__":
    main()
