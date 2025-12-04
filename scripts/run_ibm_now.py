#!/usr/bin/env python3
"""Run the variational circuit on IBM Quantum."""
import os
import json
import logging
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variational circuit (4 qubits, 5 layers)
JSON_CIRCUIT = [
    {"gate": "rz", "targets": [0], "rotation": -3.028}, {"gate": "ry", "targets": [0], "rotation": 0.036},
    {"gate": "rz", "targets": [0], "rotation": -3.030}, {"gate": "rz", "targets": [1], "rotation": -2.808},
    {"gate": "ry", "targets": [1], "rotation": 0.273}, {"gate": "rz", "targets": [1], "rotation": -2.944},
    {"gate": "x", "targets": [1], "controls": [0]},
    {"gate": "rz", "targets": [2], "rotation": 0.088}, {"gate": "ry", "targets": [2], "rotation": 0.019},
    {"gate": "rz", "targets": [2], "rotation": 0.110}, {"gate": "rz", "targets": [3], "rotation": -3.039},
    {"gate": "ry", "targets": [3], "rotation": 0.002}, {"gate": "rz", "targets": [3], "rotation": -2.976},
    {"gate": "x", "targets": [3], "controls": [1]},
    {"gate": "x", "targets": [2], "controls": [3]}, {"gate": "x", "targets": [0], "controls": [2]},
]

def build_circuit():
    qc = QuantumCircuit(4)
    for step in JSON_CIRCUIT:
        gate, target = step['gate'], step['targets'][0]
        if gate == 'rz': qc.rz(step['rotation'], target)
        elif gate == 'ry': qc.ry(step['rotation'], target)
        elif gate == 'x':
            if 'controls' in step: qc.cx(step['controls'][0], target)
            else: qc.x(target)
    qc.measure_all()
    return qc

def main():
    qc = build_circuit()
    logging.info(f"📊 Circuito: {qc.num_qubits} qubits, depth={qc.depth()}")
    
    token = os.getenv('IBM_QUANTUM_TOKEN')
    if not token:
        logging.error("❌ IBM_QUANTUM_TOKEN no encontrado")
        return
    
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    logging.info("🔌 Conectando a IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    
    logging.info("🔍 Buscando backend...")
    backend = service.least_busy(simulator=False, operational=True, min_num_qubits=4)
    logging.info(f"   Backend: {backend.name}")
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    logging.info(f"   Transpiled depth: {isa_circuit.depth()}")
    
    logging.info("📤 Enviando a IBM...")
    sampler = Sampler(backend)
    job = sampler.run([isa_circuit], shots=4096)
    logging.info(f"✅ Job ID: {job.job_id()}")
    
    with open("output/ibm_job_id.txt", "w") as f:
        f.write(job.job_id())
    
    logging.info("⏳ Esperando resultados...")
    result = job.result()
    counts = result[0].data.meas.get_counts()
    
    print("\n--- Resultados IBM Quantum ---")
    total = sum(counts.values())
    for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"|{state}⟩: {count:4d} ({count/total:.2%})")
    
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(10,5))
    states = [s for s,_ in sorted(counts.items(), key=lambda x:x[1], reverse=True)[:16]]
    probs = [counts[s]/total for s in states]
    plt.bar(states, probs, color='navy')
    plt.title(f'IBM Quantum - {backend.name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/ibm_quantum_results.png")
    logging.info("📈 output/ibm_quantum_results.png")

if __name__ == "__main__":
    main()
