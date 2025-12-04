import os
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.engines.compute_backend import IonQBackend
    from src import config
except ImportError:
    # Modo standalone
    class Config: IONQ_API_KEY = os.getenv("IONQ_API_KEY"); IONQ_BACKEND_NAME = "ionq_simulator"
    config = Config()
    IonQBackend = None

def build_harmonic_circuit(num_qubits, time_step):
    """
    Simula la evolución de un paquete de ondas usando QFT.
    Psi(t) = iQFT @ Phase(t) @ QFT @ Psi(0)
    """
    qc = QuantumCircuit(num_qubits)
    
    # 1. Inicialización: Un estado gaussiano o superposición (Onda)
    # Aproximación simple: Hadamard en todos para una onda plana, 
    # o rotaciones para un paquete. Usaremos H para máxima interferencia.
    qc.h(range(num_qubits))
    
    qc.barrier()
    
    # 2. QFT: Pasar al dominio de frecuencias (momento)
    qc.append(QFT(num_qubits, do_swaps=True).to_gate(), range(num_qubits))
    
    # 3. Evolución Temporal (Operador de Fase en el dominio de frecuencias)
    # Simula la dispersión de la onda: e^(-i * k^2 * t)
    # Aplicamos rotaciones de fase proporcionales a k (índice del qubit)
    for i in range(num_qubits):
        # Fase depende de la frecuencia (posición del qubit) y el tiempo
        theta = (time_step * (i + 1)) 
        qc.p(theta, i)
        
    # 4. iQFT: Volver al espacio real
    qc.append(QFT(num_qubits, inverse=True, do_swaps=True).to_gate(), range(num_qubits))
    
    qc.measure_all()
    return qc

def main():
    print("🌊 SIMULACIÓN HARMONIC ENGINE (QFT) 🌊")
    API_KEY = config.IONQ_API_KEY
    # if not API_KEY:
    #     print("❌ Falta IONQ_API_KEY. Ejecuta 'export IONQ_API_KEY=...'")
    #     return
    
    qc = build_harmonic_circuit(num_qubits=4, time_step=1.5)
    print(qc.draw(output='text', idle_wires=False))
    
    # Ejecución (Mock o Real)
    use_aer = True
    if API_KEY:
        try:
            if IonQBackend:
                backend = IonQBackend(api_key=API_KEY, backend_name=config.IONQ_BACKEND_NAME)
                # Transpilar para soportar compuertas complejas como QFT
                from qiskit import transpile
                qc_transpiled = transpile(qc, backend.backend)
                counts = backend.execute('run_circuit', qc_transpiled, shots=1024)
                use_aer = False
            else:
                from qiskit_ionq import IonQProvider
                provider = IonQProvider(API_KEY)
                backend = provider.get_backend("ionq_simulator")
                job = backend.run(qc, shots=1024)
                counts = job.result().get_counts()
                use_aer = False
        except Exception as e:
            print(f"⚠️ Error conectando con IonQ: {e}")
            use_aer = True

    if use_aer:
        print("\n⚠️ Usando simulador local (Aer).")
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        backend = AerSimulator()
        job = backend.run(transpile(qc, backend), shots=1024)
        counts = job.result().get_counts()
            
    print("\n📊 Espectro de Interferencia resultante:", counts)

if __name__ == "__main__":
    main()
