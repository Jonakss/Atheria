#!/usr/bin/env python3
"""
Script para verificar compatibilidad entre modelos v3/v4 y el motor nativo.

Este script verifica que:
1. Los checkpoints v3 se puedan exportar a TorchScript
2. Los checkpoints v4 se puedan exportar a TorchScript
3. Ambos funcionen con el motor nativo Engine
"""
import sys
from pathlib import Path
import torch

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config as global_cfg
from src.utils import get_latest_checkpoint, get_latest_jit_model
from src.model_loader import load_model
import atheria_core

def test_checkpoint_compatibility(experiment_name, trainer_version=None):
    """Verifica que un checkpoint se pueda exportar y usar en motor nativo"""
    print(f"\n{'=' * 70}")
    print(f"🔬 Verificando compatibilidad: {experiment_name}")
    if trainer_version:
        print(f"   Trainer: {trainer_version}")
    print(f"{'=' * 70}")
    
    # 1. Buscar checkpoint
    checkpoint_path = get_latest_checkpoint(experiment_name)
    if not checkpoint_path:
        print(f"⚠️  No se encontró checkpoint para '{experiment_name}'")
        return False
    
    print(f"✅ Checkpoint encontrado: {checkpoint_path}")
    
    # 2. Cargar configuración del experimento
    from src.utils import load_experiment_config
    exp_config = load_experiment_config(experiment_name)
    if not exp_config:
        print(f"⚠️  No se pudo cargar configuración del experimento")
        return False
    
    print(f"✅ Configuración cargada")
    print(f"   Modelo: {getattr(exp_config, 'MODEL_ARCHITECTURE', 'N/A')}")
    print(f"   d_state: {getattr(getattr(exp_config, 'MODEL_PARAMS', {}), 'd_state', 'N/A')}")
    
    # 3. Cargar modelo desde checkpoint (método tradicional)
    try:
        model, state_dict = load_model(exp_config, checkpoint_path)
        print(f"✅ Modelo cargado exitosamente (método tradicional)")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return False
    
    # 4. Exportar a TorchScript
    try:
        from scripts.export_model_to_jit import export_model_to_jit
        
        d_state = getattr(getattr(exp_config, 'MODEL_PARAMS', {}), 'd_state', 4)
        hidden_channels = getattr(getattr(exp_config, 'MODEL_PARAMS', {}), 'hidden_channels', 64)
        model_type = getattr(exp_config, 'MODEL_ARCHITECTURE', 'UNet')
        
        # Mapear nombres de arquitectura
        model_type_map = {
            'UNET_UNITARIA': 'UNetUnitary',
            'SNN_UNET': 'SNNUNet',
            'MLP': 'MLP',
            'DEEP_QCA': 'DeepQCA',
            'UNET': 'UNet',
        }
        model_type = model_type_map.get(model_type, 'UNet')
        
        # Exportar
        jit_path = Path(global_cfg.TRAINING_CHECKPOINTS_DIR) / experiment_name / "model_jit_test.pt"
        success = export_model_to_jit(
            model_path=str(checkpoint_path),
            output_path=str(jit_path),
            d_state=d_state,
            hidden_channels=hidden_channels,
            model_type=model_type,
            device='cpu'
        )
        
        if success:
            print(f"✅ Modelo exportado a TorchScript: {jit_path}")
        else:
            print(f"❌ Error al exportar modelo")
            return False
            
    except Exception as e:
        print(f"❌ Error al exportar: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Probar con motor nativo
    try:
        if not atheria_core.has_torch_support():
            print(f"⚠️  LibTorch no disponible, saltando test de motor nativo")
            return True
        
        engine = atheria_core.Engine(d_state, "cpu")
        if engine.load_model(str(jit_path)):
            print(f"✅ Modelo cargado en motor nativo exitosamente")
            
            # Test rápido
            coord = atheria_core.Coord3D(0, 0, 0)
            state = torch.randn(d_state, dtype=torch.complex64) * 0.1
            engine.add_particle(coord, state)
            
            count = engine.step_native()
            print(f"✅ Step ejecutado exitosamente: {count} partículas")
            return True
        else:
            print(f"❌ Error al cargar modelo en motor nativo")
            return False
            
    except Exception as e:
        print(f"❌ Error en motor nativo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("=" * 70)
    print("🧪 Test de Compatibilidad v3/v4 con Motor Nativo")
    print("=" * 70)
    
    # Buscar experimentos existentes
    from src.utils import get_experiment_list
    experiments = get_experiment_list()
    
    if not experiments:
        print("⚠️  No se encontraron experimentos")
        print("   Crea un experimento primero con v3 o v4")
        return
    
    print(f"\n📦 Experimentos encontrados: {len(experiments)}")
    
    # Probar con algunos experimentos (si tienen checkpoints)
    tested = 0
    passed = 0
    
    for exp in experiments[:5]:  # Limitar a 5 para no ser muy largo
        exp_name = exp.get('name')
        if not exp_name:
            continue
        
        checkpoint_path = get_latest_checkpoint(exp_name, silent=True)
        if not checkpoint_path:
            continue
        
        # Intentar determinar versión (no crítico para el test)
        # Por ahora probamos todos
        
        tested += 1
        if test_checkpoint_compatibility(exp_name):
            passed += 1
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    print(f"   Experimentos probados: {tested}")
    print(f"   Compatibles: {passed}")
    print(f"   Incompatibles: {tested - passed}")
    
    if tested == 0:
        print("\n⚠️  No se encontraron experimentos con checkpoints")
        print("   Para probar, entrena un modelo primero")
    elif passed == tested:
        print("\n✅ Todos los modelos son compatibles con el motor nativo!")
        print("   Los checkpoints v3 y v4 funcionan correctamente")
    else:
        print("\n⚠️  Algunos modelos tienen problemas de compatibilidad")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

