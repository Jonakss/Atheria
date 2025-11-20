#!/usr/bin/env python3
"""
Script de prueba para el motor nativo C++.

Convierte un modelo entrenado a TorchScript y lo prueba con el motor nativo.
"""

import os
import sys
import torch
import logging
import time
from pathlib import Path

# Configurar path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import load_model
from src.utils import load_experiment_config, get_latest_checkpoint
from src.engines.native_engine_wrapper import NativeEngineWrapper

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def export_model_to_torchscript(model, device, output_path, grid_size=64, d_state=8):
    """
    Exporta un modelo PyTorch a TorchScript.
    
    Args:
        model: Modelo PyTorch
        device: Dispositivo (torch.device)
        output_path: Ruta donde guardar el .pt
        grid_size: Tamaño del grid para crear ejemplo de entrada
        d_state: Dimensión del estado
        
    Returns:
        Ruta del archivo .pt exportado
    """
    model.eval()
    
    # Crear ejemplo de entrada: [batch=1, channels=2*d_state, H, W]
    # El modelo espera entrada concatenada [real, imag]
    example_input = torch.randn(1, 2 * d_state, grid_size, grid_size, device=device)
    
    logger.info(f"Exportando modelo a TorchScript...")
    logger.info(f"  Input shape: {example_input.shape}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Model type: {model.__class__.__name__}")
    
    try:
        # Verificar que el modelo puede hacer forward con el ejemplo de entrada
        logger.info("  Verificando forward pass...")
        with torch.no_grad():
            try:
                output = model(example_input)
                logger.info(f"  Forward pass exitoso. Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
            except Exception as forward_error:
                logger.error(f"  ❌ Error en forward pass: {forward_error}", exc_info=True)
                return None
        
        # Intentar usar torch.jit.trace primero (más compatible con modelos que tienen memoria)
        traced_model = None
        try:
            logger.info("  Intentando torch.jit.trace...")
            with torch.no_grad():
                # Para modelos con memoria (ConvLSTM), necesitamos asegurar que no haya estados persistentes
                if hasattr(model, 'reset_hidden_state'):
                    model.reset_hidden_state()
                
                # Intentar trace con ejemplo de entrada
                traced_model = torch.jit.trace(model, example_input, strict=False)
                logger.info("  ✅ torch.jit.trace exitoso")
        except Exception as trace_error:
            logger.warning(f"  torch.jit.trace falló: {trace_error}")
            # Intentar torch.jit.script como fallback (puede fallar con modelos complejos)
            try:
                logger.info("  Intentando torch.jit.script...")
                with torch.no_grad():
                    traced_model = torch.jit.script(model)
                logger.info("  ✅ torch.jit.script exitoso")
            except Exception as script_error:
                logger.error(f"  ❌ torch.jit.script también falló: {script_error}", exc_info=True)
                return None
        
        if traced_model is None:
            logger.error("  ❌ No se pudo exportar el modelo con ningún método")
            return None
        
        # Guardar modelo exportado
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        traced_model.save(output_path)
        logger.info(f"✅ Modelo exportado a: {output_path}")
        
        # Verificar que se puede cargar
        loaded_model = torch.jit.load(output_path, map_location=device)
        logger.info(f"✅ Modelo TorchScript verificado (carga exitosa)")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Error exportando modelo: {e}", exc_info=True)
        return None

def test_native_engine(experiment_name, device_str="cpu", num_steps=10):
    """
    Prueba el motor nativo C++ con un modelo real.
    
    Args:
        experiment_name: Nombre del experimento a probar
        device_str: Dispositivo ('cpu' o 'cuda')
        num_steps: Número de pasos de simulación a ejecutar
    """
    logger.info("=" * 80)
    logger.info(f"🧪 TEST: Motor Nativo C++ con Modelo Real")
    logger.info("=" * 80)
    
    # 1. Verificar que el módulo C++ está disponible
    try:
        # Si hay problemas de CUDA, forzar CPU mode desde el inicio
        import os
        if device_str == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        import atheria_core
        logger.info(f"✅ Módulo C++ importable: atheria_core")
        logger.info(f"   has_torch_support: {atheria_core.has_torch_support()}")
        
        # Si se usó CPU mode, mantenerlo
        if device_str == "cpu":
            device_str = "cpu"
            logger.info("   Usando CPU mode (forzado para evitar problemas de CUDA runtime)")
    except (ImportError, OSError, RuntimeError) as e:
        error_str = str(e)
        if '__nvJitLinkCreate' in error_str or 'libnvJitLink' in error_str:
            # Problema de CUDA runtime - intentar en CPU mode
            logger.warning(f"⚠️ Problema de CUDA runtime detectado: {error_str[:100]}")
            logger.info("   Intentando importar en CPU mode...")
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                # Reimportar atheria_core después de deshabilitar CUDA
                import importlib
                if 'atheria_core' in sys.modules:
                    del sys.modules['atheria_core']
                import atheria_core
                device_str = "cpu"  # Forzar CPU mode
                logger.info(f"✅ Módulo C++ importable en CPU mode: atheria_core")
                logger.info(f"   has_torch_support: {atheria_core.has_torch_support()}")
            except Exception as e2:
                logger.error(f"❌ Error: atheria_core no disponible incluso en CPU mode: {e2}")
                logger.error("   Compila el módulo con: python setup.py build_ext --inplace")
                return False
        else:
            logger.error(f"❌ Error: atheria_core no disponible: {e}")
            logger.error("   Compila el módulo con: python setup.py build_ext --inplace")
            return False
    
    # 2. Cargar configuración del experimento
    logger.info(f"\n📋 Cargando configuración del experimento: {experiment_name}")
    try:
        exp_config = load_experiment_config(experiment_name)
        if not exp_config:
            logger.error(f"❌ No se encontró configuración para '{experiment_name}'")
            return False
        logger.info(f"✅ Configuración cargada")
        logger.info(f"   Arquitectura: {exp_config.MODEL_ARCHITECTURE}")
        logger.info(f"   d_state: {exp_config.MODEL_PARAMS.d_state}")
        logger.info(f"   Grid size (inference): {exp_config.GRID_SIZE_INFERENCE if hasattr(exp_config, 'GRID_SIZE_INFERENCE') else 'N/A'}")
    except Exception as e:
        logger.error(f"❌ Error cargando configuración: {e}", exc_info=True)
        return False
    
    # 3. Cargar modelo entrenado
    logger.info(f"\n📦 Cargando modelo entrenado...")
    try:
        checkpoint_path = get_latest_checkpoint(experiment_name)
        if not checkpoint_path:
            logger.error(f"❌ No se encontró checkpoint para '{experiment_name}'")
            logger.error("   Entrena el modelo primero antes de probarlo")
            return False
        
        logger.info(f"   Checkpoint: {checkpoint_path}")
        model, state_dict = load_model(exp_config, checkpoint_path)
        
        if model is None:
            logger.error(f"❌ Error cargando modelo desde checkpoint")
            return False
        
        logger.info(f"✅ Modelo cargado exitosamente")
        logger.info(f"   Tipo: {model.__class__.__name__}")
        
        # Obtener parámetros
        d_state = exp_config.MODEL_PARAMS.d_state
        grid_size = getattr(exp_config, 'GRID_SIZE_INFERENCE', 128)
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}", exc_info=True)
        return False
    
    # 4. Exportar modelo a TorchScript
    logger.info(f"\n📤 Exportando modelo a TorchScript...")
    device = torch.device(device_str)
    model = model.to(device)
    
    torchscript_path = PROJECT_ROOT / "output" / "torchscript_models" / f"{experiment_name}.pt"
    exported_path = export_model_to_torchscript(
        model, 
        device, 
        str(torchscript_path),
        grid_size=grid_size,
        d_state=d_state
    )
    
    if not exported_path or not os.path.exists(exported_path):
        logger.error(f"❌ Error: No se pudo exportar el modelo")
        return False
    
    # 5. Inicializar motor nativo
    logger.info(f"\n🚀 Inicializando motor nativo C++...")
    try:
        wrapper = NativeEngineWrapper(
            grid_size=grid_size,
            d_state=d_state,
            device=device_str,
            cfg=exp_config
        )
        logger.info(f"✅ Motor nativo inicializado")
    except Exception as e:
        logger.error(f"❌ Error inicializando motor nativo: {e}", exc_info=True)
        return False
    
    # 6. Cargar modelo en el motor nativo
    logger.info(f"\n📥 Cargando modelo TorchScript en motor nativo...")
    try:
        success = wrapper.load_model(str(torchscript_path))
        if not success:
            logger.error(f"❌ Error cargando modelo en motor nativo")
            return False
        logger.info(f"✅ Modelo cargado en motor nativo")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}", exc_info=True)
        return False
    
    # 7. Agregar partículas iniciales
    logger.info(f"\n✨ Agregando partículas iniciales...")
    try:
        wrapper.add_initial_particles(num_particles=10)
        logger.info(f"✅ {wrapper.native_engine.get_matter_count()} partículas agregadas")
    except Exception as e:
        logger.error(f"❌ Error agregando partículas: {e}", exc_info=True)
        return False
    
    # 8. Ejecutar pasos de simulación
    logger.info(f"\n⏱️  Ejecutando {num_steps} pasos de simulación...")
    try:
        times = []
        particle_counts = []
        
        for step in range(num_steps):
            start_time = time.time()
            wrapper.evolve_internal_state()
            elapsed = time.time() - start_time
            
            particle_count = wrapper.native_engine.get_matter_count()
            current_step = wrapper.native_engine.get_step_count()
            
            times.append(elapsed)
            particle_counts.append(particle_count)
            
            logger.info(f"   Paso {step + 1}/{num_steps}: "
                       f"{elapsed*1000:.2f}ms, "
                       f"{particle_count} partículas, "
                       f"step_count={current_step}")
        
        avg_time = sum(times) / len(times)
        avg_particles = sum(particle_counts) / len(particle_counts)
        total_time = sum(times)
        
        logger.info(f"\n📊 Métricas de Rendimiento:")
        logger.info(f"   Tiempo promedio por paso: {avg_time*1000:.2f}ms")
        logger.info(f"   Tiempo total: {total_time:.3f}s")
        logger.info(f"   Partículas promedio: {avg_particles:.1f}")
        logger.info(f"   Throughput: {num_steps/total_time:.2f} pasos/segundo")
        
    except Exception as e:
        logger.error(f"❌ Error ejecutando simulación: {e}", exc_info=True)
        return False
    
    # 9. Verificar estado final
    logger.info(f"\n🔍 Verificando estado final...")
    try:
        final_psi = wrapper.state.psi
        if final_psi is not None:
            logger.info(f"✅ Estado cuántico disponible")
            logger.info(f"   Shape: {final_psi.shape}")
            logger.info(f"   Device: {final_psi.device}")
            logger.info(f"   Dtype: {final_psi.dtype}")
            logger.info(f"   Es complejo: {final_psi.is_complex()}")
            
            # Calcular estadísticas
            abs_values = torch.abs(final_psi)
            logger.info(f"   Min: {abs_values.min().item():.6f}")
            logger.info(f"   Max: {abs_values.max().item():.6f}")
            logger.info(f"   Mean: {abs_values.mean().item():.6f}")
        else:
            logger.warning("⚠️  Estado cuántico es None")
    except Exception as e:
        logger.error(f"❌ Error verificando estado: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TEST COMPLETADO EXITOSAMENTE")
    logger.info("=" * 80)
    
    return True

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Probar motor nativo C++ con modelo real")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Nombre del experimento a probar"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo a usar (cpu o cuda)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Número de pasos de simulación a ejecutar"
    )
    
    args = parser.parse_args()
    
    success = test_native_engine(
        experiment_name=args.experiment,
        device_str=args.device,
        num_steps=args.steps
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
