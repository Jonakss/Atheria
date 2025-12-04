#!/usr/bin/env python3
"""
Execute a JSON-defined quantum circuit on IonQ hardware.
Usage: python scripts/run_json_circuit_ionq.py
"""
import json
import os
import sys
import logging
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import IonQProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Circuit from user (Variational Quantum Circuit - 4 qubits)
JSON_CIRCUIT = [
    {"gate": "rz", "targets": [0], "rotation": -3.0280701612406444},
    {"gate": "ry", "targets": [0], "rotation": 0.03645160794258132},
    {"gate": "rz", "targets": [0], "rotation": -3.03002257843549},
    {"gate": "rz", "targets": [1], "rotation": -2.8083389719300946},
    {"gate": "ry", "targets": [1], "rotation": 0.2727651000022891},
    {"gate": "rz", "targets": [1], "rotation": -2.943951862054416},
    {"gate": "x", "targets": [1], "controls": [0]},
    {"gate": "rz", "targets": [2], "rotation": 0.08849918097256904},
    {"gate": "ry", "targets": [2], "rotation": 0.018581133335828653},
    {"gate": "rz", "targets": [2], "rotation": 0.10995396971703286},
    {"gate": "rz", "targets": [3], "rotation": -3.0388055538111898},
    {"gate": "ry", "targets": [3], "rotation": 0.002248285803944005},
    {"gate": "rz", "targets": [3], "rotation": -2.975568847852249},
    {"gate": "x", "targets": [3], "controls": [1]},
    {"gate": "rz", "targets": [1], "rotation": -0.11640157550573349},
    {"gate": "ry", "targets": [1], "rotation": 0.37355428934097284},
    {"gate": "rz", "targets": [1], "rotation": -0.14595644176006406},
    {"gate": "x", "targets": [2], "controls": [3]},
    {"gate": "x", "targets": [0], "controls": [2]},
    {"gate": "rz", "targets": [0], "rotation": 0.33136013150215193},
    {"gate": "ry", "targets": [0], "rotation": 0.1380158066749571},
    {"gate": "rz", "targets": [0], "rotation": 0.26909932494163424},
    {"gate": "x", "targets": [1], "controls": [0]},
    {"gate": "rz", "targets": [2], "rotation": -0.30448651313781694},
    {"gate": "ry", "targets": [2], "rotation": 0.2836277186870574},
    {"gate": "rz", "targets": [2], "rotation": -0.3182553648948674},
    {"gate": "rz", "targets": [3], "rotation": 0.09231761842966213},
    {"gate": "ry", "targets": [3], "rotation": 0.033817283809184924},
    {"gate": "rz", "targets": [3], "rotation": 0.03540370985865415},
    {"gate": "x", "targets": [3], "controls": [1]},
    {"gate": "rz", "targets": [1], "rotation": 3.1252287105941257},
    {"gate": "ry", "targets": [1], "rotation": 0.152500241994858},
    {"gate": "rz", "targets": [1], "rotation": 3.109662665175744},
    {"gate": "x", "targets": [2], "controls": [3]},
    {"gate": "x", "targets": [0], "controls": [2]},
    {"gate": "rz", "targets": [0], "rotation": 0.05921812355518252},
    {"gate": "ry", "targets": [0], "rotation": 0.1703963428735732},
    {"gate": "rz", "targets": [0], "rotation": 0.056488391011953354},
    {"gate": "x", "targets": [1], "controls": [0]},
    {"gate": "rz", "targets": [2], "rotation": -3.045005077617713},
    {"gate": "ry", "targets": [2], "rotation": 0.3032838702201845},
    {"gate": "rz", "targets": [2], "rotation": -3.133907483080872},
    {"gate": "rz", "targets": [3], "rotation": 0.3216755092144008},
    {"gate": "ry", "targets": [3], "rotation": 0.24490745365619643},
    {"gate": "rz", "targets": [3], "rotation": 0.28182387351989746},
    {"gate": "x", "targets": [3], "controls": [1]},
    {"gate": "rz", "targets": [1], "rotation": 2.775230052071162},
    {"gate": "ry", "targets": [1], "rotation": 0.21129862964153293},
    {"gate": "rz", "targets": [1], "rotation": 2.8865345438295087},
    {"gate": "x", "targets": [2], "controls": [3]},
    {"gate": "x", "targets": [0], "controls": [2]},
    {"gate": "rz", "targets": [0], "rotation": -0.3698447048664093},
    {"gate": "ry", "targets": [0], "rotation": 0.3077238798141477},
    {"gate": "rz", "targets": [0], "rotation": -0.2644510567188263},
    {"gate": "x", "targets": [1], "controls": [0]},
    {"gate": "rz", "targets": [2], "rotation": -2.9753456135564544},
    {"gate": "ry", "targets": [2], "rotation": 0.043207172304392014},
    {"gate": "rz", "targets": [2], "rotation": -2.9221327473693552},
    {"gate": "rz", "targets": [3], "rotation": -2.578846963244028},
    {"gate": "ry", "targets": [3], "rotation": 0.136762097477913},
    {"gate": "rz", "targets": [3], "rotation": -2.8061679025464743},
    {"gate": "x", "targets": [3], "controls": [1]},
    {"gate": "rz", "targets": [1], "rotation": -2.787211449938365},
    {"gate": "ry", "targets": [1], "rotation": 0.20226547122001673},
    {"gate": "rz", "targets": [1], "rotation": -2.741912605362483},
    {"gate": "x", "targets": [2], "controls": [3]},
    {"gate": "x", "targets": [0], "controls": [2]},
    {"gate": "rz", "targets": [0], "rotation": -0.2262718379497528},
    {"gate": "ry", "targets": [0], "rotation": 0.12086854130029662},
    {"gate": "rz", "targets": [0], "rotation": -0.07714129239320755},
    {"gate": "x", "targets": [1], "controls": [0]},
    {"gate": "rz", "targets": [2], "rotation": 2.964276315765926},
    {"gate": "ry", "targets": [2], "rotation": 0.1793185025453569},
    {"gate": "rz", "targets": [2], "rotation": 3.0399054755740833},
    {"gate": "rz", "targets": [3], "rotation": 2.9211190511756637},
    {"gate": "ry", "targets": [3], "rotation": 0.08846731483936322},
    {"gate": "rz", "targets": [3], "rotation": 3.068591082590169},
    {"gate": "x", "targets": [3], "controls": [1]},
    {"gate": "x", "targets": [2], "controls": [3]},
    {"gate": "x", "targets": [0], "controls": [2]}
]


def build_circuit_from_json(instructions):
    """Parse JSON and build Qiskit QuantumCircuit."""
    n_qubits = max(max(step['targets']) for step in instructions) + 1
    qc = QuantumCircuit(n_qubits)
    
    for step in instructions:
        gate = step['gate']
        target = step['targets'][0]
        
        if gate == 'rz':
            qc.rz(step['rotation'], target)
        elif gate == 'ry':
            qc.ry(step['rotation'], target)
        elif gate == 'rx':
            qc.rx(step['rotation'], target)
        elif gate == 'x':
            if 'controls' in step:
                control = step['controls'][0]
                qc.cx(control, target)
            else:
                qc.x(target)
        elif gate == 'h':
            qc.h(target)
    
    return qc


def main():
    logging.info("🚀 Ejecutando circuito JSON en IonQ...")
    
    # 1. Build Circuit
    qc = build_circuit_from_json(JSON_CIRCUIT)
    qc.measure_all()
    
    logging.info(f"📊 Circuito: {qc.num_qubits} qubits, Profundidad: {qc.depth()}")
    logging.info(f"   Gate Count: {sum(qc.count_ops().values())}")
    
    # Draw circuit
    print("\n--- Circuito (ASCII) ---")
    print(qc.draw(output='text', fold=120))
    
    # 2. Connect to IonQ
    api_key = os.getenv('IONQ_API_KEY')
    if not api_key:
        logging.error("❌ IONQ_API_KEY no encontrada. Ejecutando simulación local...")
        # Fallback to local simulation
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        result = simulator.run(qc, shots=4096).result()
        counts = result.get_counts()
    else:
        provider = IonQProvider(token=api_key)
        # Use simulator for testing, or 'ionq_qpu.aria-1' for real hardware
        backend_name = os.getenv('IONQ_BACKEND_NAME', 'ionq_simulator')
        backend = provider.get_backend(backend_name)
        
        logging.info(f"🔌 Backend: {backend.name}")
        
        # Transpile for IonQ
        qc_transpiled = transpile(qc, backend, optimization_level=1)
        
        # Run
        logging.info("📤 Enviando a IonQ...")
        job = backend.run(qc_transpiled, shots=4096)
        job_id = job.job_id()
        
        logging.info(f"✅ Job ID: {job_id}")
        
        with open("ionq_variational_job_id.txt", "w") as f:
            f.write(job_id)
        
        logging.info("⏳ Esperando resultados...")
        result = job.result()
        counts = result.get_counts()
    
    # 3. Display Results
    print("\n--- Resultados (Top 10) ---")
    total = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for state, count in sorted_counts[:10]:
        prob = count / total
        print(f"|{state}⟩: {count:4d} shots ({prob:.2%})")
    
    # 4. Save histogram
    output_path = "docs/40_Experiments/images/variational_circuit_results.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    states = [s for s, _ in sorted_counts[:16]]
    probs = [c/total for _, c in sorted_counts[:16]]
    
    plt.figure(figsize=(12, 6))
    plt.bar(states, probs, color='teal')
    plt.xlabel('Estado Base')
    plt.ylabel('Probabilidad')
    plt.title('Variational Quantum Circuit - IonQ Results')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"\n📈 Histograma: {output_path}")


if __name__ == "__main__":
    main()
