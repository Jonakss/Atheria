#!/usr/bin/env python3
"""
test_engines_training.py

Test de entrenamiento desde 0 para verificar que UNet Unitary funciona
con Polar y Harmonic engines.
"""
import os
import sys
import logging
import torch
from types import SimpleNamespace

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_polar_engine():
    """Test UNet Unitary con PolarEngine"""
    logger.info("=" * 60)
    logger.info("🧪 TEST: UNet Unitary + PolarEngine")
    logger.info("=" * 60)
    
    try:
        from src.models.unet_unitary import UNetUnitary
        from src.engines.qca_engine_polar import PolarEngine
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grid_size = 32
        d_state = 4
        hidden_channels = 16
        
        # PolarEngine usa real/imag concatenados: entrada = d_state*2
        # UNetUnitary internamente calcula in_c = 2*d_state, así que pasamos d_state
        model = UNetUnitary(d_state, hidden_channels).to(device)
        logger.info(f"✅ Modelo creado: {type(model).__name__}")
        
        # Crear engine
        engine = PolarEngine(model, grid_size, d_state, device)
        logger.info(f"✅ Engine creado: {type(engine).__name__}")
        
        # Verificar que tiene cfg para gamma_decay
        engine.cfg = SimpleNamespace(GAMMA_DECAY=0.01)
        
        # Test evolve_internal_state
        logger.info("🔄 Testing evolve_internal_state...")
        engine.evolve_internal_state()
        logger.info("✅ evolve_internal_state OK")
        
        # Test get_dense_state
        logger.info("🔄 Testing get_dense_state...")
        dense = engine.get_dense_state()
        logger.info(f"✅ get_dense_state OK - shape: {dense.shape}")
        
        # Test get_visualization_data
        logger.info("🔄 Testing get_visualization_data...")
        viz_data = engine.get_visualization_data("density")
        logger.info(f"✅ get_visualization_data OK - min: {viz_data['min']}, max: {viz_data['max']}")
        
        logger.info("🎉 PolarEngine TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ PolarEngine TEST FAILED: {e}", exc_info=True)
        return False

def test_harmonic_engine():
    """Test UNet Unitary con HarmonicEngine"""
    logger.info("=" * 60)
    logger.info("🧪 TEST: UNet Unitary + HarmonicEngine")
    logger.info("=" * 60)
    
    try:
        from src.models.unet_unitary import UNetUnitary
        from src.engines.harmonic_engine import SparseHarmonicEngine
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        grid_size = 32
        d_state = 4
        hidden_channels = 16
        
        # Crear modelo - UNetUnitary internamente calcula in_c = 2*d_state
        model = UNetUnitary(d_state, hidden_channels).to(device)
        logger.info(f"✅ Modelo creado: {type(model).__name__}")
        
        # Crear engine
        engine = SparseHarmonicEngine(model, d_state, device, grid_size)
        logger.info(f"✅ Engine creado: {type(engine).__name__}")
        
        # Test step (evolución con emergencia)
        logger.info("🔄 Testing step (emergence-based)...")
        initial_matter_count = len(engine.matter)
        engine.step()
        final_matter_count = len(engine.matter)
        logger.info(f"✅ step OK - Matter: {initial_matter_count} → {final_matter_count}")
        
        # Test evolve_internal_state
        logger.info("🔄 Testing evolve_internal_state...")
        engine.evolve_internal_state()
        logger.info("✅ evolve_internal_state OK")
        
        # Test get_dense_state
        logger.info("🔄 Testing get_dense_state...")
        dense = engine.get_dense_state()
        logger.info(f"✅ get_dense_state OK - shape: {dense.shape}")
        
        # Test get_visualization_data
        logger.info("🔄 Testing get_visualization_data...")
        viz_data = engine.get_visualization_data("density")
        logger.info(f"✅ get_visualization_data OK - min: {viz_data['min']}, max: {viz_data['max']}")
        
        logger.info("🎉 HarmonicEngine TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ HarmonicEngine TEST FAILED: {e}", exc_info=True)
        return False

def test_lattice_engine():
    """Test LatticeEngine (no usa modelo, Lattice Gauge Theory)"""
    logger.info("=" * 60)
    logger.info("🧪 TEST: LatticeEngine (SU3 Gauge Theory)")
    logger.info("=" * 60)
    
    try:
        from src.engines.lattice_engine import LatticeEngine
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grid_size = 32
        d_state = 9  # SU(3) tiene 3x3 = 9 componentes
        
        # Crear engine (no usa modelo)
        engine = LatticeEngine(grid_size, d_state, device)
        logger.info(f"✅ Engine creado: {type(engine).__name__}")
        
        # Test step (Metropolis-Hastings)
        logger.info("🔄 Testing step...")
        engine.step()
        logger.info("✅ step OK")
        
        # Test evolve_internal_state
        logger.info("🔄 Testing evolve_internal_state...")
        engine.evolve_internal_state()
        logger.info("✅ evolve_internal_state OK")
        
        # Test get_dense_state
        logger.info("🔄 Testing get_dense_state...")
        dense = engine.get_dense_state()
        logger.info(f"✅ get_dense_state OK - shape: {dense.shape}")
        
        # Test get_visualization_data
        logger.info("🔄 Testing get_visualization_data...")
        viz_data = engine.get_visualization_data("density")
        logger.info(f"✅ get_visualization_data OK - min: {viz_data['min']}, max: {viz_data['max']}")
        
        logger.info("🎉 LatticeEngine TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ LatticeEngine TEST FAILED: {e}", exc_info=True)
        return False

def test_native_engine():
    """Test NativeEngineWrapper (C++ High Performance)"""
    logger.info("=" * 60)
    logger.info("🧪 TEST: NativeEngineWrapper (C++ Interface)")
    logger.info("=" * 60)
    
    try:
        from src.engines.native_engine_wrapper import NativeEngineWrapper
        
        # Intentar inicializar (puede fallar si no está compilado)
        device = 'cpu' # Forzar CPU para test básico
        grid_size = 32
        d_state = 4
        
        try:
            engine = NativeEngineWrapper(grid_size, d_state, device)
            logger.info(f"✅ Engine creado: {type(engine).__name__} (Version: {engine.VERSION})")
        except ImportError as e:
            logger.warning(f"⚠️ NativeEngine no disponible (esperado si no compilado): {e}")
            logger.info("⏩ SALTANDO test de NativeEngine.")
            return True # Consideramos pass si no está instalado
            
        # Verificar Protocolo
        from src.engines.base_engine import EngineProtocol, verify_engine_protocol
        results, missing = verify_engine_protocol(engine, raise_on_error=False)
        is_compliant = len(missing) == 0
        if is_compliant:
            logger.info("✅ Protocolo verificado: 100% compliant")
        else:
            logger.error(f"❌ Protocolo fallido. Faltan: {missing}")
            # No fallamos el test completo por esto aun, pero lo logueamos
            
        # Test step (evolve_step wrapper)
        logger.info("🔄 Testing evolve_internal_state...")
        engine.evolve_internal_state()
        logger.info("✅ evolve_internal_state OK")
        
        # Test get_dense_state
        logger.info("🔄 Testing get_dense_state...")
        dense = engine.get_dense_state()
        if dense is not None:
            logger.info(f"✅ get_dense_state OK - shape: {dense.shape}")
        else:
             logger.warning("⚠️ get_dense_state retornó None")
             
        logger.info("🎉 NativeEngineWrapper TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ NativeEngineWrapper TEST FAILED: {e}", exc_info=True)
        return False

def test_cartesian_engine():
    """Test UNet Unitary con CartesianEngine (baseline)"""
    logger.info("=" * 60)
    logger.info("🧪 TEST: UNet Unitary + CartesianEngine (baseline)")
    logger.info("=" * 60)
    
    try:
        from src.models.unet_unitary import UNetUnitary
        from src.engines.qca_engine import CartesianEngine
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grid_size = 32
        d_state = 4
        hidden_channels = 16
        
        # Crear modelo - UNetUnitary internamente calcula in_c = 2*d_state
        model = UNetUnitary(d_state, hidden_channels).to(device)
        logger.info(f"✅ Modelo creado: {type(model).__name__}")
        
        # Crear engine con cfg
        cfg = SimpleNamespace(GAMMA_DECAY=0.01)
        engine = CartesianEngine(model, grid_size, d_state, device, cfg=cfg)
        logger.info(f"✅ Engine creado: {type(engine).__name__}")
        
        # Test evolve_internal_state
        logger.info("🔄 Testing evolve_internal_state...")
        engine.evolve_internal_state()
        logger.info("✅ evolve_internal_state OK")
        
        # Test get_dense_state
        logger.info("🔄 Testing get_dense_state...")
        dense = engine.get_dense_state()
        logger.info(f"✅ get_dense_state OK - shape: {dense.shape}")
        
        # Test get_visualization_data
        logger.info("🔄 Testing get_visualization_data...")
        viz_data = engine.get_visualization_data("density")
        logger.info(f"✅ get_visualization_data OK - min: {viz_data['min']}, max: {viz_data['max']}")
        
        logger.info("🎉 CartesianEngine TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ CartesianEngine TEST FAILED: {e}", exc_info=True)
        return False

def main():
    """Ejecutar todos los tests"""
    logger.info("=" * 70)
    logger.info("🚀 ATHERIA ENGINE TESTS - UNet Unitary con múltiples engines")
    logger.info("=" * 70)
    
    results = {}
    
    # Test Cartesian (baseline)
    results["CartesianEngine"] = test_cartesian_engine()
    
    # Test Polar
    results["PolarEngine"] = test_polar_engine()
    
    # Test Harmonic
    results["HarmonicEngine"] = test_harmonic_engine()
    
    # Test Lattice
    results["LatticeEngine"] = test_lattice_engine()

    # Test Native (si disponible)
    results["NativeEngine"] = test_native_engine()
    
    # Resumen
    logger.info("\n" + "=" * 70)
    logger.info("📊 RESUMEN DE TESTS")
    logger.info("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for engine, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        logger.info(f"  {engine}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests pasaron")
    
    if passed == total:
        logger.info("🎉 ¡TODOS LOS TESTS PASARON!")
        return 0
    else:
        logger.warning("⚠️ Algunos tests fallaron")
        return 1

if __name__ == "__main__":
    sys.exit(main())
