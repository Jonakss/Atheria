#!/usr/bin/env python3
"""
Benchmark comparativo de rendimiento: Motor Python vs Motor C++ Nativo

Este script ejecuta el mismo experimento con ambos motores y compara:
- Tiempo de ejecución (pasos/segundo)
- Uso de memoria
- Throughput
- Precisión de resultados
"""

import os
import sys
import torch
import time
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import psutil
import gc

# Configurar path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import load_model, create_new_model
from src.utils import load_experiment_config, get_latest_checkpoint
from src.engines.qca_engine import Aetheria_Motor, QuantumState
from src.engines.native_engine_wrapper import NativeEngineWrapper, NATIVE_AVAILABLE
from src import config as global_cfg

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Obtiene el uso de memoria actual en MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_python_engine(
    exp_config: Any,
    checkpoint_path: Optional[str],
    num_steps: int,
    grid_size: int,
    device: torch.device
) -> Dict[str, Any]:
    """
    Benchmark del motor Python (Aetheria_Motor).
    
    Returns:
        Dict con métricas de rendimiento
    """
    logger.info("🐍 Iniciando benchmark del motor Python...")
    
    # Limpiar memoria antes de empezar
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    memory_before = get_memory_usage()
    
    try:
        # Cargar modelo
        start_time = time.time()
        if checkpoint_path:
            model, _ = load_model(exp_config, checkpoint_path)
        else:
            model = create_new_model(exp_config)
        model.to(device)
        model.eval()  # Modo evaluación para benchmark
        load_time = time.time() - start_time
        
        # Inicializar motor Python
        motor = Aetheria_Motor(
            model,
            grid_size,
            exp_config.MODEL_PARAMS.d_state,
            device,
            cfg=exp_config
        )
        
        # Inicializar estado cuántico
        state = QuantumState(
            grid_size,
            exp_config.MODEL_PARAMS.d_state,
            device,
            initial_mode='complex_noise'
        )
        
        init_time = time.time() - start_time
        memory_after_init = get_memory_usage()
        
        # Warm-up (ejecutar algunos pasos para calentar)
        logger.info("   Calentando motor Python...")
        with torch.no_grad():
            for _ in range(min(10, num_steps // 10)):
                state.psi = motor.evolve_step(state.psi)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark principal
        logger.info(f"   Ejecutando {num_steps} pasos...")
        step_start = time.time()
        
        with torch.no_grad():
            for step in range(num_steps):
                state.psi = motor.evolve_step(state.psi)
                if step % (num_steps // 10) == 0 and step > 0:
                    logger.info(f"   Paso {step}/{num_steps} completado")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        step_time = time.time() - step_start
        memory_after = get_memory_usage()
        
        # Calcular métricas
        steps_per_second = num_steps / step_time if step_time > 0 else 0
        
        # Obtener energía final para comparación
        final_energy = torch.mean(torch.abs(state.psi) ** 2).item()
        
        logger.info(f"✅ Motor Python completado: {steps_per_second:.2f} pasos/seg")
        
        return {
            'engine': 'Python',
            'load_time': load_time,
            'init_time': init_time,
            'step_time': step_time,
            'total_time': time.time() - start_time,
            'steps_per_second': steps_per_second,
            'memory_before': memory_before,
            'memory_after_init': memory_after_init,
            'memory_after': memory_after,
            'memory_peak': memory_after,
            'final_energy': final_energy,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Error en benchmark Python: {e}", exc_info=True)
        return {
            'engine': 'Python',
            'success': False,
            'error': str(e)
        }

def benchmark_native_engine(
    exp_config: Any,
    checkpoint_path: Optional[str],
    num_steps: int,
    grid_size: int,
    device: torch.device
) -> Dict[str, Any]:
    """
    Benchmark del motor C++ nativo (NativeEngineWrapper).
    
    Returns:
        Dict con métricas de rendimiento
    """
    if not NATIVE_AVAILABLE:
        logger.warning("⚠️ Motor nativo no disponible. Saltando benchmark.")
        return {
            'engine': 'Native C++',
            'success': False,
            'error': 'Motor nativo no disponible'
        }
    
    logger.info("⚡ Iniciando benchmark del motor C++ nativo...")
    
    # Limpiar memoria antes de empezar
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    memory_before = get_memory_usage()
    
    try:
        # Inicializar wrapper del motor nativo
        start_time = time.time()
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        wrapper = NativeEngineWrapper(
            grid_size=grid_size,
            d_state=exp_config.MODEL_PARAMS.d_state,
            device=device_str,
            cfg=exp_config
        )
        init_time = time.time() - start_time
        memory_after_init = get_memory_usage()
        
        # Exportar modelo a TorchScript si no existe
        if checkpoint_path:
            # Crear directorio para TorchScript si no existe
            torchscript_dir = Path(global_cfg.TRAINING_CHECKPOINTS_DIR) / exp_config.EXPERIMENT_NAME / 'torchscript'
            torchscript_dir.mkdir(parents=True, exist_ok=True)
            torchscript_path = torchscript_dir / 'model.pt'
            
            if not torchscript_path.exists():
                logger.info("   Exportando modelo a TorchScript...")
                from scripts.test_native_engine import export_model_to_torchscript
                
                # Cargar modelo temporalmente para exportar
                model, _ = load_model(exp_config, checkpoint_path)
                model.to(device)
                model.eval()
                
                export_model_to_torchscript(
                    model,
                    device,
                    str(torchscript_path),
                    grid_size=grid_size,
                    d_state=exp_config.MODEL_PARAMS.d_state
                )
                
                load_time = time.time() - start_time
            else:
                load_time = time.time() - start_time
                logger.info(f"   TorchScript encontrado: {torchscript_path}")
            
            # Cargar modelo en motor nativo
            logger.info("   Cargando modelo en motor nativo...")
            wrapper.load_model(str(torchscript_path))
        else:
            logger.warning("⚠️ Sin checkpoint, el motor nativo necesita un modelo TorchScript")
            return {
                'engine': 'Native C++',
                'success': False,
                'error': 'No se proporcionó checkpoint'
            }
        
        # Inicializar estado con algunas partículas
        logger.info("   Inicializando estado...")
        # Agregar algunas partículas para que haya actividad
        import numpy as np
        for i in range(5):
            x = np.random.randint(0, grid_size)
            y = np.random.randint(0, grid_size)
            z = 0
            state_vec = torch.randn(exp_config.MODEL_PARAMS.d_state, dtype=torch.float32) * 0.1
            wrapper.native_engine.add_particle((x, y, z), state_vec.numpy())
        
        init_time = time.time() - start_time
        memory_after_init = get_memory_usage()
        
        # Warm-up
        logger.info("   Calentando motor nativo...")
        for _ in range(min(10, num_steps // 10)):
            wrapper.evolve_step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark principal
        logger.info(f"   Ejecutando {num_steps} pasos...")
        step_start = time.time()
        
        for step in range(num_steps):
            wrapper.evolve_step()
            if step % (num_steps // 10) == 0 and step > 0:
                logger.info(f"   Paso {step}/{num_steps} completado")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        step_time = time.time() - step_start
        memory_after = get_memory_usage()
        
        # Calcular métricas
        steps_per_second = num_steps / step_time if step_time > 0 else 0
        
        # Obtener energía final para comparación
        final_energy = torch.mean(torch.abs(wrapper.state.psi) ** 2).item() if wrapper.state.psi is not None else 0.0
        
        logger.info(f"✅ Motor nativo completado: {steps_per_second:.2f} pasos/seg")
        
        return {
            'engine': 'Native C++',
            'load_time': load_time,
            'init_time': init_time,
            'step_time': step_time,
            'total_time': time.time() - start_time,
            'steps_per_second': steps_per_second,
            'memory_before': memory_before,
            'memory_after_init': memory_after_init,
            'memory_after': memory_after,
            'memory_peak': memory_after,
            'final_energy': final_energy,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Error en benchmark nativo: {e}", exc_info=True)
        return {
            'engine': 'Native C++',
            'success': False,
            'error': str(e)
        }

def generate_report(results: Dict[str, Any], output_path: Path):
    """Genera un reporte Markdown con los resultados del benchmark"""
    
    python_result = results.get('python')
    native_result = results.get('native')
    
    report = []
    report.append("# 🏎️ Benchmark: Motor Python vs Motor C++ Nativo\n")
    report.append(f"**Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Experimento:** {results.get('experiment_name', 'N/A')}\n")
    report.append(f"**Pasos ejecutados:** {results.get('num_steps', 'N/A')}\n")
    report.append(f"**Grid Size:** {results.get('grid_size', 'N/A')}\n")
    report.append(f"**Device:** {results.get('device', 'N/A')}\n\n")
    
    report.append("## 📊 Resultados\n\n")
    
    # Tabla comparativa
    report.append("| Métrica | Python | Native C++ | Mejora |\n")
    report.append("|---------|--------|------------|--------|\n")
    
    if python_result and python_result.get('success') and native_result and native_result.get('success'):
        # Throughput
        py_sps = python_result['steps_per_second']
        native_sps = native_result['steps_per_second']
        speedup = native_sps / py_sps if py_sps > 0 else 0
        report.append(f"| **Pasos/segundo** | {py_sps:.2f} | {native_sps:.2f} | {speedup:.2f}x |\n")
        
        # Tiempo total
        py_time = python_result['total_time']
        native_time = native_result['total_time']
        time_speedup = py_time / native_time if native_time > 0 else 0
        report.append(f"| **Tiempo total (s)** | {py_time:.2f} | {native_time:.2f} | {time_speedup:.2f}x |\n")
        
        # Memoria
        py_mem = python_result['memory_peak']
        native_mem = native_result['memory_peak']
        mem_ratio = py_mem / native_mem if native_mem > 0 else 0
        report.append(f"| **Memoria pico (MB)** | {py_mem:.2f} | {native_mem:.2f} | {mem_ratio:.2f}x |\n")
        
        # Tiempo de carga
        py_load = python_result.get('load_time', 0)
        native_load = native_result.get('load_time', 0)
        if py_load > 0 and native_load > 0:
            load_speedup = py_load / native_load
            report.append(f"| **Tiempo de carga (s)** | {py_load:.2f} | {native_load:.2f} | {load_speedup:.2f}x |\n")
        
        # Energía final (comparación de precisión)
        py_energy = python_result.get('final_energy', 0)
        native_energy = native_result.get('final_energy', 0)
        energy_diff = abs(py_energy - native_energy) / py_energy * 100 if py_energy > 0 else 0
        report.append(f"| **Energía final** | {py_energy:.6f} | {native_energy:.6f} | {energy_diff:.2f}% diff |\n")
        
        report.append("\n## 📈 Análisis\n\n")
        
        if speedup > 1:
            report.append(f"✅ **El motor nativo es {speedup:.2f}x más rápido** que el motor Python.\n\n")
        elif speedup < 1:
            report.append(f"⚠️ **El motor Python es {1/speedup:.2f}x más rápido** que el motor nativo.\n\n")
        else:
            report.append("⚖️ **Rendimiento similar** entre ambos motores.\n\n")
        
        if mem_ratio > 1:
            report.append(f"💾 **El motor Python usa {mem_ratio:.2f}x más memoria** que el motor nativo.\n\n")
        elif mem_ratio < 1:
            report.append(f"💾 **El motor nativo usa {1/mem_ratio:.2f}x más memoria** que el motor Python.\n\n")
        
        if energy_diff < 1:
            report.append(f"✅ **Precisión excelente**: Diferencia de energía < 1%\n\n")
        elif energy_diff < 5:
            report.append(f"⚠️ **Precisión aceptable**: Diferencia de energía {energy_diff:.2f}%\n\n")
        else:
            report.append(f"❌ **Precisión problemática**: Diferencia de energía {energy_diff:.2f}%\n\n")
    
    else:
        if not python_result or not python_result.get('success'):
            report.append("❌ **Motor Python:** Error\n")
            if python_result:
                report.append(f"   Error: {python_result.get('error', 'Desconocido')}\n\n")
        
        if not native_result or not native_result.get('success'):
            report.append("❌ **Motor nativo:** Error\n")
            if native_result:
                report.append(f"   Error: {native_result.get('error', 'Desconocido')}\n\n")
    
    report.append("\n## 🔧 Detalles Técnicos\n\n")
    report.append("### Motor Python\n")
    if python_result and python_result.get('success'):
        report.append(f"- Tiempo de inicialización: {python_result.get('init_time', 0):.2f}s\n")
        report.append(f"- Tiempo de pasos: {python_result.get('step_time', 0):.2f}s\n")
        report.append(f"- Memoria inicial: {python_result.get('memory_before', 0):.2f} MB\n")
        report.append(f"- Memoria pico: {python_result.get('memory_peak', 0):.2f} MB\n")
    
    report.append("\n### Motor C++ Nativo\n")
    if native_result and native_result.get('success'):
        report.append(f"- Tiempo de inicialización: {native_result.get('init_time', 0):.2f}s\n")
        report.append(f"- Tiempo de pasos: {native_result.get('step_time', 0):.2f}s\n")
        report.append(f"- Memoria inicial: {native_result.get('memory_before', 0):.2f} MB\n")
        report.append(f"- Memoria pico: {native_result.get('memory_peak', 0):.2f} MB\n")
    
    # Guardar reporte
    with open(output_path, 'w') as f:
        f.write(''.join(report))
    
    logger.info(f"📄 Reporte guardado en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark comparativo: Motor Python vs Motor C++ Nativo")
    parser.add_argument('--experiment', type=str, required=True, help='Nombre del experimento')
    parser.add_argument('--steps', type=int, default=100, help='Número de pasos a ejecutar (default: 100)')
    parser.add_argument('--warmup', type=int, default=10, help='Pasos de warm-up (default: 10)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda) - default: auto')
    parser.add_argument('--output', type=str, default=None, help='Ruta del reporte (default: benchmark_report.md)')
    
    args = parser.parse_args()
    
    # Determinar device
    if args.device:
        device = torch.device(args.device)
    else:
        device = global_cfg.get_device()
    
    logger.info("=" * 70)
    logger.info("🏎️  BENCHMARK: Motor Python vs Motor C++ Nativo")
    logger.info("=" * 70)
    logger.info(f"Experimento: {args.experiment}")
    logger.info(f"Device: {device}")
    logger.info(f"Pasos: {args.steps}")
    
    # Cargar configuración del experimento
    exp_config = load_experiment_config(args.experiment)
    if not exp_config:
        logger.error(f"❌ No se encontró configuración para el experimento '{args.experiment}'")
        sys.exit(1)
    
    # Obtener checkpoint
    checkpoint_path = get_latest_checkpoint(args.experiment)
    if not checkpoint_path:
        logger.warning("⚠️ No se encontró checkpoint. El motor nativo puede no funcionar correctamente.")
    
    grid_size = getattr(exp_config, 'GRID_SIZE_TRAINING', 64)
    
    results = {
        'experiment_name': args.experiment,
        'num_steps': args.steps,
        'grid_size': grid_size,
        'device': str(device),
        'python': None,
        'native': None
    }
    
    # Benchmark Python
    logger.info("\n" + "=" * 70)
    python_result = benchmark_python_engine(
        exp_config,
        checkpoint_path,
        args.steps,
        grid_size,
        device
    )
    results['python'] = python_result
    
    # Limpiar memoria entre benchmarks
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    time.sleep(2)  # Dar tiempo para limpiar
    
    # Benchmark Native
    logger.info("\n" + "=" * 70)
    native_result = benchmark_native_engine(
        exp_config,
        checkpoint_path,
        args.steps,
        grid_size,
        device
    )
    results['native'] = native_result
    
    # Generar reporte
    output_path = Path(args.output) if args.output else Path(f"benchmark_report_{args.experiment}.md")
    generate_report(results, output_path)
    
    # Resumen en consola
    logger.info("\n" + "=" * 70)
    logger.info("📊 RESUMEN")
    logger.info("=" * 70)
    
    if python_result.get('success') and native_result.get('success'):
        py_sps = python_result['steps_per_second']
        native_sps = native_result['steps_per_second']
        speedup = native_sps / py_sps if py_sps > 0 else 0
        
        logger.info(f"Python:   {py_sps:.2f} pasos/seg")
        logger.info(f"Native:   {native_sps:.2f} pasos/seg")
        logger.info(f"Speedup:  {speedup:.2f}x")
    
    logger.info(f"\n📄 Reporte completo: {output_path}")

if __name__ == "__main__":
    main()

