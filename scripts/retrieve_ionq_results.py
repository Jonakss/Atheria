import os
import sys
import logging
import matplotlib.pyplot as plt
from qiskit_ionq import IonQProvider

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("📥 Recuperando resultados de IonQ...")
    
    # 1. Leer Job ID
    try:
        with open("ionq_job_id.txt", "r") as f:
            job_id = f.read().strip()
    except FileNotFoundError:
        logging.error("❌ No se encontró el archivo 'ionq_job_id.txt'.")
        return

    logging.info(f"🆔 Job ID: {job_id}")
    
    # 2. Conectar a IonQ
    api_key = os.getenv('IONQ_API_KEY')
    if not api_key:
        logging.error("❌ IONQ_API_KEY no encontrada.")
        return
        
    provider = IonQProvider(token=api_key)
    # No necesitamos especificar backend para retrieve_job
    
    try:
        job = provider.get_backend('ionq_simulator').retrieve_job(job_id)
    except Exception as e:
        # Fallback: try to find backend from job? No, provider.retrieve_job is not standard.
        # We need a backend to retrieve.
        # Let's try to get it via the backend used.
        backend = provider.get_backend('ionq_simulator') # Assuming simulator was used
        job = backend.retrieve_job(job_id)
    
    # 3. Obtener Resultados
    status = job.status()
    logging.info(f"   Estado del Job: {status.name}")
    
    if status.name not in ['DONE', 'COMPLETED']:
        logging.warning("⚠️ El trabajo aún no está completo.")
        return

    result = job.result()
    counts = result.get_counts()
    
    logging.info(f"📊 Resultados (Counts):")
    # Sort by probability
    total_shots = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    # Print Top 10
    print("\nTop 10 Estados más probables:")
    print("-" * 30)
    print(f"{'Estado':<10} | {'Cuentas':<10} | {'Probabilidad':<10}")
    print("-" * 30)
    
    top_states = []
    top_probs = []
    
    for state, count in sorted_counts[:10]:
        prob = count / total_shots
        print(f"{state:<10} | {count:<10} | {prob:.4f}")
        top_states.append(state)
        top_probs.append(prob)
        
    # Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.bar(top_states, top_probs, color='purple')
    plt.xlabel('Estado Base')
    plt.ylabel('Probabilidad')
    plt.title(f'Resultados IonQ (Job: {job_id})')
    plt.xticks(rotation=45)
    
    output_path = "docs/40_Experiments/images/ionq_results_histogram.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"\n📈 Histograma guardado en: {output_path}")

if __name__ == "__main__":
    main()
