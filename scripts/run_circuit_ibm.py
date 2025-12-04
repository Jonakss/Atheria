#!/usr/bin/env python3
"""
Execute quantum circuits on IBM Quantum hardware.
Supports both the Strongly Entangling model and custom JSON circuits.

Usage: 
  python scripts/run_circuit_ibm.py --mode model
  python scripts/run_circuit_ibm.py --mode json
"""
import os
import sys
import json
import argparse
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_model_circuit():
    """Build circuit from trained StronglyEntangling model."""
    import torch
    from scripts.experiment_advanced_ansatz import StronglyEntanglingConv2d
    
    model_path = "output/models/quantum_fastforward_final.pt"
    if not os.path.exists(model_path):
        logging.error(f"❌ Modelo no encontrado: {model_path}")
        return None
    
    cp = torch.load(model_path, map_location='cpu')
    config = cp['config']
    
    model = StronglyEntanglingConv2d(
        grid_size=config['grid_size'],
        n_layers=config['n_layers'],
        device='cpu'
    )
    model.load_state_dict(cp['model_state_dict'])
    
    qc = model.export_circuit()
    qc.measure_all()
    return qc


def build_json_circuit():
    """Build circuit from JSON (variational circuit)."""
    from qiskit import QuantumCircuit
    
    # Same circuit as run_json_circuit_ionq.py
    json_path = "output/variational_circuit.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            instructions = json.load(f)
    else:
        # Embedded default
        instructions = [
            {"gate": "rz", "targets": [0], "rotation": -3.028},
            {"gate": "ry", "targets": [0], "rotation": 0.036},
            {"gate": "rz", "targets": [0], "rotation": -3.030},
            {"gate": "x", "targets": [1], "controls": [0]},
            # ... simplified for demo
        ]
    
    n_qubits = max(max(step['targets']) for step in instructions) + 1
    qc = QuantumCircuit(n_qubits)
    
    for step in instructions:
        gate = step['gate']
        target = step['targets'][0]
        
        if gate == 'rz':
            qc.rz(step['rotation'], target)
        elif gate == 'ry':
            qc.ry(step['rotation'], target)
        elif gate == 'x':
            if 'controls' in step:
                qc.cx(step['controls'][0], target)
            else:
                qc.x(target)
    
    qc.measure_all()
    return qc


def run_on_ibm(qc, shots=4096):
    """Execute circuit on IBM Quantum."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    # Check for IBM token
    token = os.getenv('IBM_QUANTUM_TOKEN')
    if not token:
        logging.error("❌ IBM_QUANTUM_TOKEN no encontrado.")
        logging.info("   Obtén tu token en: https://quantum.ibm.com/")
        return None
    
    # Connect to IBM
    logging.info("🔌 Conectando a IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    
    # Get least busy backend with enough qubits
    logging.info("🔍 Buscando backend disponible...")
    backend = service.least_busy(
        simulator=False,
        operational=True,
        min_num_qubits=qc.num_qubits
    )
    logging.info(f"   Backend seleccionado: {backend.name}")
    
    # Transpile for target backend
    logging.info("⚙️ Transpilando circuito...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    
    logging.info(f"   Profundidad original: {qc.depth()}")
    logging.info(f"   Profundidad transpilada: {isa_circuit.depth()}")
    
    # Execute
    logging.info("📤 Enviando a IBM Quantum...")
    sampler = Sampler(backend)
    job = sampler.run([isa_circuit], shots=shots)
    
    job_id = job.job_id()
    logging.info(f"✅ Job enviado! ID: {job_id}")
    
    with open("output/ibm_job_id.txt", "w") as f:
        f.write(job_id)
    
    # Wait for results
    logging.info("⏳ Esperando resultados (puede tomar varios minutos)...")
    result = job.result()
    
    # Extract counts
    counts = result[0].data.meas.get_counts()
    return counts, job_id, backend.name


def run_local_simulation(qc, shots=4096):
    """Fallback to local Aer simulation."""
    from qiskit_aer import AerSimulator
    
    logging.info("🖥️ Ejecutando simulación local (AerSimulator)...")
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, "local", "aer_simulator"


def main():
    parser = argparse.ArgumentParser(description='Run circuits on IBM Quantum')
    parser.add_argument('--mode', choices=['model', 'json', 'local'], default='model',
                       help='Circuit source: model (trained), json (custom), local (simulation)')
    parser.add_argument('--shots', type=int, default=4096, help='Number of shots')
    args = parser.parse_args()
    
    logging.info(f"🚀 IBM Quantum Runner - Mode: {args.mode}")
    
    # Build circuit
    if args.mode == 'json':
        qc = build_json_circuit()
    else:
        qc = build_model_circuit()
    
    if qc is None:
        return
    
    logging.info(f"📊 Circuito: {qc.num_qubits} qubits, {qc.depth()} profundidad")
    
    # Execute
    if args.mode == 'local':
        counts, job_id, backend_name = run_local_simulation(qc, args.shots)
    else:
        result = run_on_ibm(qc, args.shots)
        if result is None:
            logging.info("⚠️ Fallback a simulación local...")
            counts, job_id, backend_name = run_local_simulation(qc, args.shots)
        else:
            counts, job_id, backend_name = result
    
    # Display results
    print(f"\n--- Resultados ({backend_name}) ---")
    total = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for state, count in sorted_counts[:10]:
        prob = count / total
        print(f"|{state}⟩: {count:4d} shots ({prob:.2%})")
    
    # Save histogram
    output_path = f"output/ibm_results_{job_id[:8]}.png"
    states = [s for s, _ in sorted_counts[:16]]
    probs = [c/total for _, c in sorted_counts[:16]]
    
    plt.figure(figsize=(12, 6))
    plt.bar(states, probs, color='darkblue')
    plt.xlabel('Estado Base')
    plt.ylabel('Probabilidad')
    plt.title(f'IBM Quantum Results - {backend_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"\n📈 Histograma: {output_path}")


if __name__ == "__main__":
    main()
