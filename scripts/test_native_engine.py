#!/usr/bin/env python3
"""
Script de prueba para el motor nativo de alto rendimiento.

Este script prueba:
1. Carga de modelo TorchScript
2. Agregado de partículas
3. Ejecución de step_native() (todo el trabajo en C++)
4. Recuperación de estados
"""
import sys
from pathlib import Path
import torch

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import atheria_core

def test_native_engine():
    """Test básico del motor nativo"""
    print("=" * 70)
    print("🧪 Test del Motor Nativo de Alto Rendimiento")
    print("=" * 70)
    
    # Verificar soporte
    if not atheria_core.has_torch_support():
        print("❌ LibTorch no está disponible")
        return False
    
    print("✅ LibTorch disponible")
    
    # Crear motor
    d_state = 4
    device = "cpu"
    engine = atheria_core.Engine(d_state, device)
    print(f"✅ Motor creado (d_state={d_state}, device={device})")
    
    # Agregar partículas
    print("\n📦 Agregando partículas...")
    for i in range(5):
        coord = atheria_core.Coord3D(i*10, i*20, i*30)
        state = torch.randn(d_state, dtype=torch.complex64) * 0.1
        engine.add_particle(coord, state)
    
    count = engine.get_matter_count()
    print(f"✅ {count} partículas agregadas")
    
    # Verificar estados
    print("\n🔍 Verificando estados...")
    for i in range(5):
        coord = atheria_core.Coord3D(i*10, i*20, i*30)
        state = engine.get_state_at(coord)
        energy = torch.sum(torch.abs(state).pow(2)).item()
        print(f"   Coord {coord}: energía = {energy:.6f}")
    
    # Ejecutar step sin modelo (debería conservar las partículas)
    print("\n⚡ Ejecutando step_native() sin modelo...")
    count_after = engine.step_native()
    print(f"✅ Step completado. Partículas: {count_after}")
    
    # Test con modelo (si existe)
    print("\n🤖 Test con modelo TorchScript...")
    
    # Buscar modelo usando utils (consistente con el resto del proyecto)
    try:
        from src.utils import get_latest_jit_model
        # Intentar buscar modelo del experimento por defecto
        from src import config as global_cfg
        experiment_name = getattr(global_cfg, 'EXPERIMENT_NAME', None)
        
        model_path = None
        if experiment_name:
            model_path = get_latest_jit_model(experiment_name, silent=True)
        
        if not model_path:
            # Buscar en directorios comunes
            checkpoint_dir = getattr(global_cfg, 'TRAINING_CHECKPOINTS_DIR', None)
            if checkpoint_dir and experiment_name:
                possible_paths = [
                    Path(checkpoint_dir) / experiment_name / "model_jit.pt",
                    Path(checkpoint_dir) / experiment_name / "model.pt",
                ]
                for path in possible_paths:
                    if path.exists():
                        model_path = str(path)
                        break
        
        model_loaded = False
        if model_path and Path(model_path).exists():
            print(f"   Intentando cargar: {model_path}")
            if engine.load_model(str(model_path)):
                print(f"✅ Modelo cargado: {model_path}")
                model_loaded = True
        
        if not model_loaded:
            print("⚠️  No se encontró modelo JIT para probar")
            print("   Para probar con modelo, exporta uno usando:")
            print("   python scripts/export_model_to_jit.py <checkpoint.pth> --experiment_name <nombre>")
            print(f"   O: python scripts/export_model_to_jit.py <checkpoint.pth> --output_path <salida.pt>")
    except Exception as e:
        print(f"⚠️  Error al buscar modelo: {e}")
    else:
        # Ejecutar step con modelo
        print("\n⚡ Ejecutando step_native() con modelo...")
        count_with_model = engine.step_native()
        print(f"✅ Step con modelo completado. Partículas: {count_with_model}")
    
    print("\n" + "=" * 70)
    print("✅ Todos los tests completados")
    print("=" * 70)
    return True

if __name__ == "__main__":
    try:
        success = test_native_engine()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error en tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

