#!/usr/bin/env python3
"""
Script para exportar modelos entrenados a formato TorchScript (JIT).

Este script toma un modelo entrenado (.pth) y lo exporta como TorchScript (.pt)
para ser usado en el motor C++ de alto rendimiento.
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
import argparse

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def export_model_to_jit(model_path, output_path=None, experiment_name=None, 
                        d_state=4, hidden_channels=64, model_type='UNet', 
                        input_shape=(3, 3), device='cpu'):
    """
    Exporta un modelo a formato TorchScript.
    
    Args:
        model_path: Ruta al modelo entrenado (.pth)
        output_path: Ruta de salida para el modelo JIT (.pt). Si None, se usa experiment_name
        experiment_name: Nombre del experimento (para determinar ruta de salida)
        d_state: Dimensión del estado cuántico
        hidden_channels: Canales ocultos del modelo
        model_type: Tipo de modelo ('UNet', 'UNetUnitary', 'DeepQCA', etc.)
        input_shape: Forma de entrada (H, W) - por defecto 3x3 para patches
        device: Dispositivo ('cpu' o 'cuda')
    """
    # Determinar ruta de salida si no se proporciona
    if output_path is None:
        if experiment_name:
            from src import config as global_cfg
            import os
            # Guardar en directorio de checkpoints del experimento
            output_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, experiment_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "model_jit.pt")
        else:
            raise ValueError("Debe proporcionar output_path o experiment_name")
    
    print(f"📦 Exportando modelo a TorchScript...")
    print(f"   Modelo: {model_path}")
    print(f"   Salida: {output_path}")
    print(f"   Tipo: {model_type}")
    print(f"   d_state: {d_state}, hidden_channels: {hidden_channels}")
    print(f"   Input shape: {input_shape}")
    
    # Importar clase del modelo
    from src.model_loader import MODEL_MAP
    
    if model_type not in MODEL_MAP:
        raise ValueError(f"Tipo de modelo '{model_type}' no encontrado. "
                        f"Opciones: {list(MODEL_MAP.keys())}")
    
    # Crear instancia del modelo
    model_class = MODEL_MAP[model_type]
    model = model_class(d_state=d_state, hidden_channels=hidden_channels)
    
    # Cargar pesos entrenados
    checkpoint = torch.load(model_path, map_location=device)
    
    # Manejar diferentes formatos de checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    # Crear tensor de ejemplo para tracing
    # El modelo espera: [batch, 2*d_state, H, W]
    batch_size = 8  # Batch de ejemplo
    example_input = torch.randn(batch_size, 2 * d_state, input_shape[0], input_shape[1],
                               device=device, dtype=torch.float32)
    
    print(f"   Tensor de ejemplo: {example_input.shape}")
    
    # Exportar usando torch.jit.trace
    print("   Traceando modelo...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input, strict=False)
        
        # Guardar modelo JIT
        traced_model.save(output_path)
        print(f"✅ Modelo exportado exitosamente a: {output_path}")
        
        # Verificar que el modelo se puede cargar
        print("   Verificando modelo exportado...")
        loaded_model = torch.jit.load(output_path, map_location=device)
        test_output = loaded_model(example_input)
        print(f"✅ Modelo verificado. Salida: {test_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al exportar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Exporta modelos entrenados a TorchScript')
    parser.add_argument('model_path', type=str, help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--output_path', type=str, default=None, 
                       help='Ruta de salida (.pt). Si no se proporciona, usa --experiment_name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Nombre del experimento (para determinar ruta automática)')
    parser.add_argument('--d_state', type=int, default=4, help='Dimensión del estado (default: 4)')
    parser.add_argument('--hidden_channels', type=int, default=64, 
                       help='Canales ocultos (default: 64)')
    parser.add_argument('--model_type', type=str, default='UNet',
                       choices=['UNet', 'UNetUnitary', 'DeepQCA', 'MLP'],
                       help='Tipo de modelo (default: UNet)')
    parser.add_argument('--input_height', type=int, default=3,
                       help='Altura de entrada (default: 3 para patch 3x3)')
    parser.add_argument('--input_width', type=int, default=3,
                       help='Ancho de entrada (default: 3 para patch 3x3)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Dispositivo (default: cpu)')
    
    args = parser.parse_args()
    
    # Verificar que el modelo existe
    if not Path(args.model_path).exists():
        print(f"❌ Error: Modelo no encontrado: {args.model_path}")
        sys.exit(1)
    
    # Crear directorio de salida si no existe
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Exportar modelo
    success = export_model_to_jit(
        model_path=args.model_path,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        d_state=args.d_state,
        hidden_channels=args.hidden_channels,
        model_type=args.model_type,
        input_shape=(args.input_height, args.input_width),
        device=args.device
    )
    
    if success:
        print("\n✅ Exportación completada exitosamente!")
        print(f"   El modelo está listo para usar en el motor C++")
        sys.exit(0)
    else:
        print("\n❌ Error en la exportación")
        sys.exit(1)

if __name__ == "__main__":
    main()

