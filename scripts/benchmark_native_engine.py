#!/usr/bin/env python3
"""
Benchmark del motor nativo C++ con optimizaciones.
Mide rendimiento real y compara con motor Python.
"""
import sys
import os
import time
import logging
import argparse
import torch

# Agregar proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configurar logging
logging.basicConfig(
    level=logging.WARNING,  # Reducir verbosidad para benchmarks
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def benchmark_native_engine_steps(grid_size=256, d_state=8, num_steps=1000, device='cpu'):
    """Benchmark del motor nativo ejecutando pasos."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Motor Nativo C++ - Solo Pasos (sin conversión)")
    print(f"{'='*60}")
    print(f"Grid: {grid_size}x{grid_size}, d_state: {d_state}, Device: {device}")
    print(f"Pasos a ejecutar: {num_steps}")
    
    try:
        from src.engines.native_engine_wrapper import NativeEngineWrapper
        
        # Crear wrapper
        wrapper = NativeEngineWrapper(grid_size=grid_size, d_state=d_state, device=device)
        print("✅ Wrapper creado")
        
        # Sin modelo, solo medir overhead
        print("\n⚠️  Nota: Sin modelo cargado, solo medimos overhead de evolve_internal_state()")
        
        # Calentar
        for _ in range(10):
            try:
                wrapper.evolve_internal_state()
            except:
                pass
        
        # Benchmark
        start_time = time.time()
        steps_executed = 0
        
        for i in range(num_steps):
            try:
                wrapper.evolve_internal_state()
                steps_executed += 1
            except Exception as e:
                if "Modelo no cargado" not in str(e):
                    print(f"⚠️  Error en paso {i}: {e}")
                break
        
        elapsed_time = time.time() - start_time
        
        if steps_executed > 0:
            steps_per_second = steps_executed / elapsed_time
            avg_time_per_step = elapsed_time / steps_executed * 1000  # ms
            
            print(f"\n📊 RESULTADOS:")
            print(f"   Pasos ejecutados: {steps_executed:,}")
            print(f"   Tiempo total: {elapsed_time:.3f} segundos")
            print(f"   Pasos/segundo: {steps_per_second:,.1f}")
            print(f"   Tiempo/paso: {avg_time_per_step:.3f} ms")
            
            return {
                'steps_executed': steps_executed,
                'elapsed_time': elapsed_time,
                'steps_per_second': steps_per_second,
                'avg_time_per_step': avg_time_per_step
            }
        else:
            print("❌ No se ejecutaron pasos")
            return None
            
    except ImportError as e:
        print(f"❌ Módulo nativo no disponible: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_conversion(grid_size=256, d_state=8, device='cpu', use_roi=False):
    """Benchmark de conversión disperso→denso."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Conversión Disperso→Denso")
    print(f"{'='*60}")
    print(f"Grid: {grid_size}x{grid_size}, d_state: {d_state}, Device: {device}")
    print(f"ROI: {'Habilitada (128x128)' if use_roi else 'Deshabilitada (grid completo)'}")
    
    try:
        from src.engines.native_engine_wrapper import NativeEngineWrapper
        
        # Crear wrapper
        wrapper = NativeEngineWrapper(grid_size=grid_size, d_state=d_state, device=device)
        print("✅ Wrapper creado")
        
        # Definir ROI si se solicita
        roi = None
        if use_roi:
            roi_size = grid_size // 2
            roi = (grid_size // 4, grid_size // 4, grid_size // 4 + roi_size, grid_size // 4 + roi_size)
            print(f"   ROI: ({roi[0]}, {roi[1]}) - ({roi[2]}, {roi[3]})")
            coords_in_roi = (roi[2] - roi[0]) * (roi[3] - roi[1])
            coords_total = grid_size * grid_size
            reduction = (1 - coords_in_roi / coords_total) * 100
            print(f"   Coordenadas: {coords_in_roi:,} / {coords_total:,} ({reduction:.1f}% reducción)")
        
        # Calentar
        wrapper.get_dense_state(roi=roi)
        
        # Benchmark conversión
        num_iterations = 10
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            psi = wrapper.get_dense_state(roi=roi)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        conversions_per_second = 1.0 / avg_time
        
        print(f"\n📊 RESULTADOS ({num_iterations} iteraciones):")
        print(f"   Tiempo promedio: {avg_time*1000:.2f} ms")
        print(f"   Tiempo mínimo: {min_time*1000:.2f} ms")
        print(f"   Tiempo máximo: {max_time*1000:.2f} ms")
        print(f"   Conversiones/segundo: {conversions_per_second:.1f}")
        
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'conversions_per_second': conversions_per_second,
            'roi': roi
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_combined(grid_size=256, d_state=8, device='cpu', num_steps=100, visualize_every=10):
    """Benchmark combinado: pasos + conversión periódica."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Combinado (Pasos + Conversión)")
    print(f"{'='*60}")
    print(f"Grid: {grid_size}x{grid_size}, d_state: {d_state}, Device: {device}")
    print(f"Pasos totales: {num_steps}, Visualizar cada: {visualize_every} pasos")
    
    try:
        from src.engines.native_engine_wrapper import NativeEngineWrapper
        
        # Crear wrapper
        wrapper = NativeEngineWrapper(grid_size=grid_size, d_state=d_state, device=device)
        print("✅ Wrapper creado")
        
        start_time = time.time()
        steps_time = 0
        conversion_time = 0
        steps_executed = 0
        conversions = 0
        
        for i in range(num_steps):
            # Ejecutar paso
            step_start = time.time()
            try:
                wrapper.evolve_internal_state()
                steps_executed += 1
            except:
                pass
            steps_time += time.time() - step_start
            
            # Convertir cada visualize_every pasos (simulando live feed)
            if (i + 1) % visualize_every == 0:
                conv_start = time.time()
                try:
                    wrapper.get_dense_state()
                    conversions += 1
                except:
                    pass
                conversion_time += time.time() - conv_start
        
        total_time = time.time() - start_time
        
        if steps_executed > 0:
            steps_per_second = steps_executed / steps_time if steps_time > 0 else 0
            frames_per_second = conversions / conversion_time if conversion_time > 0 else 0
            effective_fps = conversions / total_time if total_time > 0 else 0
            
            print(f"\n📊 RESULTADOS:")
            print(f"   Tiempo total: {total_time:.3f} segundos")
            print(f"   Tiempo en pasos: {steps_time:.3f} segundos ({steps_time/total_time*100:.1f}%)")
            print(f"   Tiempo en conversión: {conversion_time:.3f} segundos ({conversion_time/total_time*100:.1f}%)")
            print(f"   Pasos ejecutados: {steps_executed:,}")
            print(f"   Conversiones: {conversions}")
            print(f"   Pasos/segundo: {steps_per_second:,.1f}")
            print(f"   Conversiones/segundo: {frames_per_second:.1f}")
            print(f"   FPS efectivo: {effective_fps:.1f}")
            
            return {
                'total_time': total_time,
                'steps_time': steps_time,
                'conversion_time': conversion_time,
                'steps_executed': steps_executed,
                'conversions': conversions,
                'steps_per_second': steps_per_second,
                'frames_per_second': frames_per_second,
                'effective_fps': effective_fps
            }
        else:
            print("❌ No se ejecutaron pasos")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Benchmark del motor nativo C++')
    parser.add_argument('--grid-size', type=int, default=256, help='Tamaño del grid')
    parser.add_argument('--d-state', type=int, default=8, help='Dimensión del estado')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Dispositivo')
    parser.add_argument('--steps', type=int, default=1000, help='Número de pasos para benchmark')
    parser.add_argument('--test', type=str, choices=['steps', 'conversion', 'combined', 'all'], 
                        default='all', help='Tipo de benchmark a ejecutar')
    parser.add_argument('--roi', action='store_true', help='Usar ROI para conversión')
    parser.add_argument('--visualize-every', type=int, default=10, 
                        help='Visualizar cada N pasos (para test combined)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BENCHMARK DEL MOTOR NATIVO C++")
    print("="*60)
    print(f"\nConfiguración:")
    print(f"  Grid: {args.grid_size}x{args.grid_size}")
    print(f"  d_state: {args.d_state}")
    print(f"  Device: {args.device}")
    print(f"  Test: {args.test}")
    
    results = {}
    
    if args.test in ['steps', 'all']:
        results['steps'] = benchmark_native_engine_steps(
            grid_size=args.grid_size,
            d_state=args.d_state,
            num_steps=args.steps,
            device=args.device
        )
    
    if args.test in ['conversion', 'all']:
        results['conversion_full'] = benchmark_conversion(
            grid_size=args.grid_size,
            d_state=args.d_state,
            device=args.device,
            use_roi=False
        )
        if args.roi:
            results['conversion_roi'] = benchmark_conversion(
                grid_size=args.grid_size,
                d_state=args.d_state,
                device=args.device,
                use_roi=True
            )
    
    if args.test in ['combined', 'all']:
        results['combined'] = benchmark_combined(
            grid_size=args.grid_size,
            d_state=args.d_state,
            device=args.device,
            num_steps=args.steps,
            visualize_every=args.visualize_every
        )
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    if 'steps' in results and results['steps']:
        print(f"\n✅ Pasos/segundo (sin conversión): {results['steps']['steps_per_second']:,.1f}")
    
    if 'conversion_full' in results and results['conversion_full']:
        print(f"✅ Conversión completa: {results['conversion_full']['avg_time']*1000:.2f} ms")
        if 'conversion_roi' in results and results['conversion_roi']:
            speedup = results['conversion_full']['avg_time'] / results['conversion_roi']['avg_time']
            print(f"✅ Conversión ROI: {results['conversion_roi']['avg_time']*1000:.2f} ms ({speedup:.2f}x más rápido)")
    
    if 'combined' in results and results['combined']:
        print(f"✅ FPS efectivo (con visualización): {results['combined']['effective_fps']:.1f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
