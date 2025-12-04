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

def build_polar_circuit():
    """
    Demostración de lógica Polar: Manipulación de Fase vs Magnitud.
    """
    qc = QuantumCircuit(2)
    
    # 1. Definir Magnitud (Amplitud)
    # Rotamos el Qubit 1 para tener cierta probabilidad (Magnitud)
    qc.ry(1.5, 1) # ~50% prob
    
    qc.barrier()
    
    # 2. Manipulación de Fase (Polar)
    # Usamos el Qubit 0 para inyectar una fase global relativa en el Qubit 1
    # sin cambiar su magnitud (probabilidad de medir 1).
    qc.h(0) 
    
    # Phase Kickback / Controlled Phase
    # Esto cambia la fase relativa del sistema |11> vs |10>
    qc.cp(3.14159 / 2, 0, 1) # Rotación de fase PI/2
    
    qc.h(0) # Interferencia para detectar el cambio de fase
    
    # Si medimos Qubit 0, sabremos sobre la fase.
    # Si medimos Qubit 1, sabremos sobre la magnitud.
    qc.measure_all()
    
    return qc

def main():
    print("🧭 SIMULACIÓN POLAR ENGINE (PHASE/MAGNITUDE) 🧭")
    API_KEY = config.IONQ_API_KEY
    # if not API_KEY:
    #     print("❌ Falta IONQ_API_KEY")
    #     return

    qc = build_polar_circuit()
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

    print("\n📊 Resultados:")
    print(counts)
    print("Interpretación:")
    print("- Qubit 0 (Derecha): Información de Fase (0=Fase 0, 1=Fase Pi)")
    print("- Qubit 1 (Izquierda): Información de Magnitud (Probabilidad de existencia)")

if __name__ == "__main__":
    main()
