import torch
import os
import sys
import logging

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_advanced_ansatz import StronglyEntanglingConv2d

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("🚀 Iniciando Despliegue de Modelo Cuántico...")
    
    # Paths
    checkpoint_path = "checkpoints/advanced_ansatz_model.pt"
    output_dir = "models"
    output_path = os.path.join(output_dir, "quantum_fastforward_final.pt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return
    
    # Load Checkpoint
    logging.info(f"📂 Cargando checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = cp['model_state_dict']
    final_fidelity = cp.get('final_fidelity', 0.0)
    
    logging.info(f"   Fidelidad del modelo: {final_fidelity:.6f}")
    
    if final_fidelity < 0.9:
        logging.warning("⚠️ La fidelidad es baja (<0.9). ¿Seguro que quieres desplegar?")
    
    # Reconstruct Model to verify
    # Infer params from state dict
    # weights shape: [n_layers, n_qubits, 3]
    weights = state_dict['weights']
    n_layers, n_qubits, _ = weights.shape
    grid_size = int(2**(n_qubits/2))
    
    logging.info(f"   Arquitectura detectada: Grid {grid_size}x{grid_size}, {n_layers} Capas")
    
    model = StronglyEntanglingConv2d(grid_size=grid_size, n_layers=n_layers, device='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Export QASM for documentation/frontend
    qc = model.export_circuit()
    try:
        import qiskit.qasm2
        qasm_str = qiskit.qasm2.export(qc)
    except:
        try:
            qasm_str = qc.qasm()
        except:
            qasm_str = "// QASM Export Failed"
            
    # Save Production Artifact
    deploy_payload = {
        'model_state_dict': model.state_dict(),
        'config': {
            'grid_size': grid_size,
            'n_layers': n_layers,
            'ansatz': 'StronglyEntangling'
        },
        'qasm': qasm_str,
        'metadata': {
            'fidelity': final_fidelity,
            'description': 'Quantum Fast Forward Operator (1M Steps)',
            'version': '1.0.0'
        }
    }
    
    torch.save(deploy_payload, output_path)
    logging.info(f"✅ Modelo desplegado exitosamente en: {output_path}")
    
    # Save QASM to separate file for easy reading
    qasm_path = os.path.join(output_dir, "quantum_fastforward.qasm")
    with open(qasm_path, "w") as f:
        f.write(qasm_str)
    logging.info(f"📄 QASM guardado en: {qasm_path}")

if __name__ == "__main__":
    main()
