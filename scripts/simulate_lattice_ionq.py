import os
import sys
from qiskit import QuantumCircuit

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.engines.compute_backend import IonQBackend
    from src import config
except ImportError:
    class Config: IONQ_API_KEY = os.getenv("IONQ_API_KEY"); IONQ_BACKEND_NAME = "ionq_simulator"
    config = Config()
    IonQBackend = None

def build_lattice_circuit():
    """
    Simula una celda unitaria de Lattice Gauge Theory (Plaqueta).
    4 Qubits de Datos (Links) + 1 Qubit de Ancilla (Medidor de Flujo).
    Operador Plaqueta: U_p = X_1 * X_2 * X_3 * X_4
    """
    # 0,1,2,3: Links (Campos), 4: Ancilla (Syndrome)
    qc = QuantumCircuit(5, 1) 
    
    # 1. Estado Inicial de los Links (Cold Start = 0)
    # Aplicamos algo de "ruido térmico"
    qc.rx(0.5, [0, 1, 2, 3])
    
    qc.barrier()
    
    # 2. Medición de la Plaqueta (Evolución de Gauge)
    # Entrelazamos los 4 links con el ancilla para proyectar el flujo
    qc.h(4) # Preparar ancilla en superposición
    
    # CNOTs desde ancilla a los links (Z2 Gauge interaction)
    qc.cx(4, 0)
    qc.cx(4, 1)
    qc.cx(4, 2)
    qc.cx(4, 3)
    
    qc.h(4) # Cerrar base X
    
    # Medir el flujo de la plaqueta
    qc.measure(4, 0)
    
    return qc

def main():
    print("🕸️ SIMULACIÓN LATTICE ENGINE (GAUGE THEORY) 🕸️")
    API_KEY = config.IONQ_API_KEY
    # if not API_KEY:
    #     print("❌ Falta IONQ_API_KEY")
    #     return

    qc = build_lattice_circuit()
    print(qc.draw(output='text', idle_wires=False))

    use_aer = True
    if API_KEY:
        try:
            if IonQBackend:
                backend = IonQBackend(api_key=API_KEY, backend_name=config.IONQ_BACKEND_NAME)
                counts = backend.execute('run_circuit', qc, shots=1024)
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

    print("\n📊 Resultado de Medición de Flujo (0=Conservado, 1=Excitado):")
    print(counts)
    if counts.get('1', 0) > counts.get('0', 0):
        print("⚠️ Flux Loop Detected (Excitación de Gauge)")
    else:
        print("✅ Gauge Symmetry Preserved")

if __name__ == "__main__":
    main()
