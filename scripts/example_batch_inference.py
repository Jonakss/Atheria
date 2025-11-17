#!/usr/bin/env python3
"""
Ejemplo de uso del BatchInferenceEngine para inferencia masiva.

Este script demuestra cómo ejecutar múltiples simulaciones en paralelo
usando batching de PyTorch.

Uso:
    python scripts/example_batch_inference.py --experiment UNET_32ch_D5_LR2e-5 --num_sims 100 --steps 1000
"""

import argparse
import sys
import os
import time
import logging

# Agregar raíz del proyecto al path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.batch_inference_engine import create_batch_engine_from_experiment, BatchInferenceEngine
from src import config as global_cfg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description='Ejemplo de inferencia batch masiva')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Nombre del experimento a cargar'
    )
    parser.add_argument(
        '--num_sims',
        type=int,
        default=32,
        help='Número de simulaciones a ejecutar en paralelo (default: 32)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Número de pasos a evolucionar (default: 1000)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamaño de batch para inferencia (default: 32)'
    )
    parser.add_argument(
        '--initial_mode',
        type=str,
        default='complex_noise',
        choices=['complex_noise', 'random', 'zeros'],
        help='Modo de inicialización (default: complex_noise)'
    )
    parser.add_argument(
        '--report_interval',
        type=int,
        default=100,
        help='Reportar estadísticas cada N pasos (default: 100)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Batch Inference Engine - Ejemplo de Uso")
    print("=" * 60)
    print(f"Experimento: {args.experiment}")
    print(f"Número de simulaciones: {args.num_sims}")
    print(f"Pasos a evolucionar: {args.steps}")
    print(f"Tamaño de batch: {args.batch_size}")
    print(f"Modo inicial: {args.initial_mode}")
    print("=" * 60)
    
    try:
        # Crear engine desde experimento
        print("\n[1/3] Cargando modelo y creando engine...")
        engine = create_batch_engine_from_experiment(
            exp_name=args.experiment,
            batch_size=args.batch_size
        )
        print(f"✅ Engine creado: {args.num_sims} simulaciones preparadas")
        
        # Inicializar estados
        print("\n[2/3] Inicializando estados cuánticos...")
        engine.initialize_states(
            num_simulations=args.num_sims,
            initial_mode=args.initial_mode
        )
        print(f"✅ {args.num_sims} estados inicializados")
        
        # Evolucionar
        print(f"\n[3/3] Evolucionando {args.steps} pasos...")
        start_time = time.time()
        
        for step in range(args.steps):
            engine.evolve_batch(steps=1)
            
            # Reportar estadísticas periódicamente
            if (step + 1) % args.report_interval == 0:
                stats = engine.get_batch_statistics()
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                sims_per_sec = steps_per_sec * args.num_sims
                
                print(f"\n--- Paso {step + 1}/{args.steps} ---")
                print(f"Tiempo transcurrido: {elapsed:.2f}s")
                print(f"Velocidad: {steps_per_sec:.2f} pasos/s")
                print(f"Throughput: {sims_per_sec:.2f} simulaciones/s")
                print(f"Energía promedio: {stats['avg_energy']:.4f} ± {stats['std_energy']:.4f}")
                print(f"Entropía promedio: {stats['avg_entropy']:.4f} ± {stats['std_entropy']:.4f}")
        
        # Estadísticas finales
        total_time = time.time() - start_time
        final_stats = engine.get_batch_statistics()
        
        print("\n" + "=" * 60)
        print("Resultados Finales")
        print("=" * 60)
        print(f"Tiempo total: {total_time:.2f}s")
        print(f"Pasos completados: {args.steps}")
        print(f"Simulaciones: {args.num_sims}")
        print(f"Throughput total: {args.num_sims * args.steps / total_time:.2f} simulaciones/s")
        print(f"\nEstadísticas agregadas:")
        print(f"  Energía: {final_stats['avg_energy']:.4f} ± {final_stats['std_energy']:.4f}")
        print(f"    Min: {final_stats['min_energy']:.4f}")
        print(f"    Max: {final_stats['max_energy']:.4f}")
        print(f"  Entropía: {final_stats['avg_entropy']:.4f} ± {final_stats['std_entropy']:.4f}")
        print("=" * 60)
        
        # Ejemplo: acceder a un estado específico
        print("\nEjemplo: Accediendo a estado específico...")
        state_0 = engine.get_state(0)
        print(f"Estado 0 - Forma: {state_0.psi.shape}")
        print(f"Estado 0 - Energía: {torch.sum(state_0.psi.abs().pow(2)).item():.4f}")
        
        print("\n✅ Inferencia batch completada exitosamente!")
        
    except Exception as e:
        logging.error(f"Error durante inferencia batch: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

