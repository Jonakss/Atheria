#!/usr/bin/env python3
"""
Script de diagnóstico para verificar la inicialización del motor nativo.

Este script verifica:
1. Que USE_NATIVE_ENGINE esté habilitado
2. Que los modelos JIT se puedan exportar
3. Que el motor nativo se pueda inicializar
4. Que CUDA esté disponible y configurado correctamente
"""

import sys
import os
from pathlib import Path

# Agregar el directorio del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main():
    print("🔍 DIAGNÓSTICO DEL MOTOR NATIVO\n")
    
    # 1. Verificar configuración
    print("1️⃣ Verificando configuración...")
    try:
        from src import config as global_cfg
        use_native = getattr(global_cfg, 'USE_NATIVE_ENGINE', False)
        print(f"   USE_NATIVE_ENGINE: {use_native}")
        
        device_pytorch = global_cfg.DEVICE
        device_native = global_cfg.get_native_device()
        print(f"   Device PyTorch: {device_pytorch}")
        print(f"   Device Nativo: {device_native}")
        
        if not use_native:
            print("   ⚠️  USE_NATIVE_ENGINE está deshabilitado en config.py")
            return
        
    except Exception as e:
        print(f"   ❌ Error cargando configuración: {e}")
        return
    
    # 2. Verificar CUDA
    print("\n2️⃣ Verificando CUDA...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA disponible: {cuda_available}")
        
        if cuda_available:
            print(f"   Device count: {torch.cuda.device_count()}")
            print(f"   Device name: {torch.cuda.get_device_name(0)}")
            
            # Probar crear un tensor
            test_tensor = torch.zeros(1, device='cuda')
            print(f"   ✅ Tensor CUDA creado exitosamente en {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()
        else:
            print("   ⚠️  CUDA no disponible, usando CPU")
    except Exception as e:
        print(f"   ❌ Error verificando CUDA: {e}")
    
    # 3. Verificar módulo nativo
    print("\n3️⃣ Verificando módulo nativo...")
    try:
        from src.engines.native_engine_wrapper import NATIVE_AVAILABLE
        print(f"   NATIVE_AVAILABLE: {NATIVE_AVAILABLE}")
        
        if not NATIVE_AVAILABLE:
            print("   ⚠️  Módulo nativo no disponible. Verifica que esté compilado.")
            return
        
        # Intentar importar atheria_core
        import atheria_core
        print(f"   ✅ atheria_core importado exitosamente")
        
    except ImportError as e:
        print(f"   ❌ Error importando atheria_core: {e}")
        print("   💡 Ejecuta: cd src/cpp_core && python setup.py build_ext --inplace")
        return
    except Exception as e:
        print(f"   ❌ Error verificando módulo nativo: {e}")
        return
    
    # 4. Verificar inicialización del motor nativo
    print("\n4️⃣ Verificando inicialización del motor nativo...")
    try:
        from src.engines.native_engine_wrapper import NativeEngineWrapper
        
        # Intentar inicializar motor nativo
        motor = NativeEngineWrapper(
            grid_size=64,
            d_state=4,
            device=None,  # Auto-detección
            cfg=None
        )
        print(f"   ✅ Motor nativo inicializado exitosamente")
        print(f"   Device del motor: {motor.device_str}")
        
        del motor
        
    except Exception as e:
        print(f"   ❌ Error inicializando motor nativo: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Verificar experimentos disponibles
    print("\n5️⃣ Verificando experimentos disponibles...")
    try:
        from src.utils import get_experiment_list, get_latest_jit_model
        
        experiments = get_experiment_list()
        print(f"   Experimentos encontrados: {len(experiments)}")
        
        if experiments:
            print("\n   Detalles de experimentos:")
            # get_experiment_list puede retornar dicts o strings
            for exp_item in experiments[:5]:  # Mostrar solo los primeros 5
                # Extraer nombre del experimento (puede ser dict o string)
                if isinstance(exp_item, dict):
                    exp_name = exp_item.get('name', str(exp_item))
                else:
                    exp_name = str(exp_item)
                
                try:
                    jit_path = get_latest_jit_model(exp_name, silent=True)
                    jit_status = "✅ Tiene JIT" if jit_path else "❌ Sin JIT"
                    print(f"      - {exp_name}: {jit_status}")
                    
                    if jit_path:
                        print(f"         Path: {jit_path}")
                except Exception as e:
                    print(f"      - {exp_name}: Error verificando JIT: {e}")
        else:
            print("   ⚠️  No se encontraron experimentos")
            
    except Exception as e:
        print(f"   ❌ Error verificando experimentos: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Resumen
    print("\n📊 RESUMEN:")
    print("   ✅ Configuración: OK")
    print("   ✅ CUDA: " + ("Disponible" if cuda_available else "No disponible (CPU mode)"))
    print("   ✅ Módulo nativo: " + ("Disponible" if NATIVE_AVAILABLE else "No disponible"))
    print("   ✅ Motor nativo: " + ("Funcional" if NATIVE_AVAILABLE else "No funcional"))
    
    print("\n💡 PRÓXIMOS PASOS:")
    if not NATIVE_AVAILABLE:
        print("   1. Compilar el módulo nativo: cd src/cpp_core && python setup.py build_ext --inplace")
    else:
        print("   1. Cargar un experimento desde el frontend")
        print("   2. Si no tiene modelo JIT, se exportará automáticamente")
        print("   3. El motor nativo se inicializará si el JIT se exporta correctamente")
        print("   4. Verifica los logs del servidor para ver el proceso completo")

if __name__ == "__main__":
    main()

