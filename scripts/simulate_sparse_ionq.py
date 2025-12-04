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

def build_sparse_circuit(steps=3):
    """
    Simula interacción Materia-Vacío (Disipación).
    Qubit 0: Materia (Sparse Particle)
    Qubit 1..N: Fluctuaciones del Vacío
    """
    qc = QuantumCircuit(steps + 1)
    
    # 1. Crear Partícula (Materia)
    qc.x(0) # Estado excitado |1>
    
    # 2. Evolución temporal con el Vacío
    for i in range(steps):
        env_qubit = i + 1
        
        # Interacción débil con una fluctuación del vacío
        # Partial SWAP o Amplitude Damping
        theta = 0.3 # Fuerza de acoplamiento con el vacío
        
        # Simula intercambio de energía: Materia <-> Vacío
        qc.rxx(theta, 0, env_qubit)
        qc.ryy(theta, 0, env_qubit)
        qc.rzz(theta, 0, env_qubit)
        
        # El "vacío" se lleva la información (no medimos los env_qubits todavía)
        qc.barrier()

    # 3. Solo medimos la materia para ver si sobrevivió
    qc.measure_all() 
    return qc

def main():
    print("🌌 SIMULACIÓN SPARSE ENGINE (OPEN SYSTEMS) 🌌")
    API_KEY = config.IONQ_API_KEY
    # if not API_KEY:
    #     print("❌ Falta IONQ_API_KEY")
    #     return

    qc = build_sparse_circuit(steps=3)
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

    print("\n📊 Estado final (Bit 0 es la materia):")
    print(counts)
    
    # Analizar supervivencia
    survival_prob = 0
    total_shots = sum(counts.values())
    for state, count in counts.items():
        # Qiskit usa little-endian, el qubit 0 es el último bit
        if state[-1] == '1': 
            survival_prob += count
    
    print(f"Probabilidad de supervivencia de la partícula: {survival_prob/total_shots:.2%}")

if __name__ == "__main__":
    main()
