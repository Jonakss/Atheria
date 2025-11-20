#!/usr/bin/env python3
"""
Script para limpiar modelos TorchScript defectuosos y checkpoints.
Verifica que los modelos .pt se puedan cargar correctamente.
"""
import os
import sys
import logging
import torch
import json
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Agregar directorio raíz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as global_cfg

def check_torchscript_model(model_path: Path) -> bool:
    """Verifica si un modelo TorchScript se puede cargar correctamente."""
    try:
        # Intentar cargar en CPU primero
        model = torch.jit.load(str(model_path), map_location='cpu')
        model.eval()
        
        # Intentar hacer un forward dummy para verificar que funciona
        with torch.no_grad():
            # Crear input dummy (necesitamos saber el tamaño)
            # Por defecto, asumimos [1, 2*d_state, 64, 64]
            dummy_input = torch.randn(1, 16, 64, 64)  # 2*d_state con d_state=8
            
            try:
                output = model(dummy_input)
                # Si es tupla (ConvLSTM), verificar el primer elemento
                if isinstance(output, tuple):
                    if len(output) == 0:
                        logger.warning(f"  ⚠️ Modelo devuelve tupla vacía: {model_path.name}")
                        return False
                    output = output[0]
                
                # Verificar que el output es un tensor válido
                if not isinstance(output, torch.Tensor):
                    logger.warning(f"  ⚠️ Modelo no devuelve tensor: {model_path.name}")
                    return False
                    
                logger.info(f"  ✅ Modelo válido: {model_path.name}")
                return True
            except Exception as e:
                logger.warning(f"  ⚠️ Error en forward pass: {model_path.name} - {str(e)[:100]}")
                return False
    except Exception as e:
        logger.error(f"  ❌ Error cargando modelo: {model_path.name} - {str(e)[:100]}")
        return False

def check_checkpoint(checkpoint_path: Path) -> bool:
    """Verifica si un checkpoint se puede cargar correctamente."""
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        
        # Verificar que tiene estructura válida
        if isinstance(checkpoint, dict):
            # Debe tener al menos 'model_state_dict' o 'state_dict'
            has_state = 'model_state_dict' in checkpoint or 'state_dict' in checkpoint
            if not has_state and len(checkpoint) == 0:
                logger.warning(f"  ⚠️ Checkpoint vacío: {checkpoint_path.name}")
                return False
        elif not isinstance(checkpoint, dict):
            # Si no es dict, asumir que es directamente el state_dict
            if checkpoint is None:
                logger.warning(f"  ⚠️ Checkpoint None: {checkpoint_path.name}")
                return False
        
        logger.info(f"  ✅ Checkpoint válido: {checkpoint_path.name}")
        return True
    except Exception as e:
        logger.error(f"  ❌ Error cargando checkpoint: {checkpoint_path.name} - {str(e)[:100]}")
        return False

def cleanup_models():
    """Limpia modelos TorchScript defectuosos."""
    torchscript_dir = PROJECT_ROOT / "output" / "torchscript_models"
    
    if not torchscript_dir.exists():
        logger.info(f"📁 Directorio de modelos TorchScript no existe: {torchscript_dir}")
        return
    
    logger.info(f"\n🔍 Verificando modelos TorchScript en: {torchscript_dir}")
    
    model_files = list(torchscript_dir.glob("*.pt"))
    if not model_files:
        logger.info("  No se encontraron modelos .pt")
        return
    
    logger.info(f"  Encontrados {len(model_files)} modelos")
    
    valid_models = []
    invalid_models = []
    
    for model_path in model_files:
        if check_torchscript_model(model_path):
            valid_models.append(model_path)
        else:
            invalid_models.append(model_path)
    
    logger.info(f"\n📊 Resumen:")
    logger.info(f"  ✅ Modelos válidos: {len(valid_models)}")
    logger.info(f"  ❌ Modelos defectuosos: {len(invalid_models)}")
    
    if invalid_models:
        logger.info(f"\n🗑️  Eliminando {len(invalid_models)} modelos defectuosos...")
        for model_path in invalid_models:
            try:
                model_path.unlink()
                logger.info(f"  ✅ Eliminado: {model_path.name}")
            except Exception as e:
                logger.error(f"  ❌ Error eliminando {model_path.name}: {e}")

def cleanup_checkpoints():
    """Limpia checkpoints defectuosos."""
    checkpoint_dir = PROJECT_ROOT / "output" / "training_checkpoints"
    
    if not checkpoint_dir.exists():
        logger.info(f"📁 Directorio de checkpoints no existe: {checkpoint_dir}")
        return
    
    logger.info(f"\n🔍 Verificando checkpoints en: {checkpoint_dir}")
    
    checkpoint_files = list(checkpoint_dir.glob("**/*.pth"))
    if not checkpoint_files:
        logger.info("  No se encontraron checkpoints .pth")
        return
    
    logger.info(f"  Encontrados {len(checkpoint_files)} checkpoints")
    
    valid_checkpoints = []
    invalid_checkpoints = []
    
    for checkpoint_path in checkpoint_files:
        if check_checkpoint(checkpoint_path):
            valid_checkpoints.append(checkpoint_path)
        else:
            invalid_checkpoints.append(checkpoint_path)
    
    logger.info(f"\n📊 Resumen:")
    logger.info(f"  ✅ Checkpoints válidos: {len(valid_checkpoints)}")
    logger.info(f"  ❌ Checkpoints defectuosos: {len(invalid_checkpoints)}")
    
    if invalid_checkpoints:
        logger.info(f"\n🗑️  Eliminando {len(invalid_checkpoints)} checkpoints defectuosos...")
        for checkpoint_path in invalid_checkpoints:
            try:
                checkpoint_path.unlink()
                logger.info(f"  ✅ Eliminado: {checkpoint_path.name}")
            except Exception as e:
                logger.error(f"  ❌ Error eliminando {checkpoint_path.name}: {e}")

def delete_all_checkpoints():
    """Elimina todos los checkpoints (solicitado por el usuario)."""
    checkpoint_dir = PROJECT_ROOT / "output" / "training_checkpoints"
    
    if not checkpoint_dir.exists():
        logger.info(f"📁 Directorio de checkpoints no existe: {checkpoint_dir}")
        return
    
    logger.info(f"\n🗑️  Eliminando TODOS los checkpoints en: {checkpoint_dir}")
    
    checkpoint_files = list(checkpoint_dir.glob("**/*.pth"))
    
    if not checkpoint_files:
        logger.info("  No se encontraron checkpoints para eliminar")
        return
    
    logger.info(f"  Encontrados {len(checkpoint_files)} checkpoints")
    
    for checkpoint_path in checkpoint_files:
        try:
            checkpoint_path.unlink()
            logger.info(f"  ✅ Eliminado: {checkpoint_path.name}")
        except Exception as e:
            logger.error(f"  ❌ Error eliminando {checkpoint_path.name}: {e}")
    
    logger.info(f"\n✅ Eliminados {len(checkpoint_files)} checkpoints")

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Limpiar modelos TorchScript y checkpoints defectuosos")
    parser.add_argument("--delete-all-checkpoints", action="store_true",
                       help="Eliminar TODOS los checkpoints (no solo los defectuosos)")
    parser.add_argument("--skip-models", action="store_true",
                       help="Saltar verificación de modelos")
    parser.add_argument("--skip-checkpoints", action="store_true",
                       help="Saltar verificación de checkpoints")
    
    args = parser.parse_args()
    
    logger.info("🧹 Iniciando limpieza de modelos y checkpoints...")
    
    # Verificar y limpiar modelos TorchScript
    if not args.skip_models:
        cleanup_models()
    
    # Eliminar todos los checkpoints si se solicita
    if args.delete_all_checkpoints:
        delete_all_checkpoints()
    elif not args.skip_checkpoints:
        # Solo verificar y eliminar checkpoints defectuosos
        cleanup_checkpoints()
    
    logger.info("\n✅ Limpieza completada")

if __name__ == "__main__":
    main()

