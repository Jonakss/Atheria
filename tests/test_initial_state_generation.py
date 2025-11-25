#!/usr/bin/env python3
"""
Script para verificar que el estado inicial denso se genera correctamente
y que las partículas emergen correctamente de él.
"""
import sys
import os
import torch
import logging

# Añadir el directorio del proyecto al path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_initial_state():
    """Prueba que el estado inicial denso se genera correctamente."""
    print("🧪 VERIFICACIÓN DEL ESTADO INICIAL DENSO")
    print("="*80)
    
    from src.engines.qca_engine import QuantumState
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_size = 256
    d_state = 8
    initial_mode = 'complex_noise'
    
    print(f"📊 Configuración: grid_size={grid_size}, d_state={d_state}, device={device}, initial_mode={initial_mode}")
    
    # Crear estado inicial denso
    print(f"\n🔄 Generando estado inicial denso con QuantumState...")
    try:
        state = QuantumState(grid_size, d_state, device, initial_mode=initial_mode)
        psi = state.psi
        print(f"✅ Estado inicial denso creado exitosamente")
    except Exception as e:
        print(f"❌ Error creando estado inicial: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verificar estadísticas del estado denso
    print(f"\n📊 ESTADÍSTICAS DEL ESTADO DENSO:")
    psi_abs = psi.abs()
    psi_abs_sq = psi_abs.pow(2)
    
    print(f"  Shape: {psi.shape}")
    print(f"  Min abs: {psi_abs.min().item():.6e}")
    print(f"  Max abs: {psi_abs.max().item():.6e}")
    print(f"  Mean abs: {psi_abs.mean().item():.6e}")
    print(f"  Std abs: {psi_abs.std().item():.6e}")
    print(f"  Min abs²: {psi_abs_sq.min().item():.6e}")
    print(f"  Max abs²: {psi_abs_sq.max().item():.6e}")
    print(f"  Mean abs²: {psi_abs_sq.mean().item():.6e}")
    
    # Verificar si el estado tiene valores significativos
    max_abs = psi_abs.max().item()
    if max_abs < 1e-10:
        print(f"\n❌ CRÍTICO: El estado inicial denso está VACÍO (max abs={max_abs:.6e})")
        print(f"❌ Esto significa que QuantumState no está generando estado correctamente")
        return False
    
    # Calcular umbral que usaría _initialize_native_state_from_dense
    psi_abs_sq_max = psi_abs_sq.max().item()
    threshold = max(psi_abs_sq_max * 0.01, 1e-6)  # 1% del máximo, mínimo 1e-6
    
    print(f"\n📊 UMBRAL DE DETECCIÓN:")
    print(f"  Umbral usado: {threshold:.6e} (1% de max abs² = {psi_abs_sq_max:.6e})")
    
    # Contar cuántas células tienen densidad significativa
    print(f"\n🔍 Contando células con densidad significativa...")
    significant_cells = 0
    total_cells = grid_size * grid_size
    
    for y in range(grid_size):
        for x in range(grid_size):
            cell_density = psi_abs_sq[0, y, x, :].sum().item()
            if cell_density > threshold:
                significant_cells += 1
    
    percentage = (significant_cells / total_cells) * 100.0
    print(f"  Células significativas: {significant_cells}/{total_cells} ({percentage:.2f}%)")
    
    if significant_cells == 0:
        print(f"\n❌ CRÍTICO: Ninguna célula tiene densidad significativa.")
        print(f"❌ El umbral ({threshold:.6e}) es demasiado alto para el estado inicial.")
        print(f"❌ O el estado inicial tiene valores muy pequeños.")
        return False
    elif significant_cells < total_cells * 0.01:  # Menos del 1%
        print(f"⚠️ ADVERTENCIA: Muy pocas células significativas ({significant_cells}/{total_cells})")
        print(f"⚠️ El estado inicial puede no ser suficiente para propagación")
    
    # Probar conversión a motor nativo
    print(f"\n" + "="*80)
    print("PRUEBA: Convertir estado denso a motor nativo")
    print("="*80)
    
    try:
        import atheria_core
        from src.engines.native_engine_wrapper import NativeEngineWrapper
        
        print(f"🔄 Creando NativeEngineWrapper...")
        # Crear un cfg simple
        class SimpleConfig:
            INITIAL_STATE_MODE_INFERENCE = initial_mode
        
        cfg = SimpleConfig()
        
        wrapper = NativeEngineWrapper(grid_size, d_state, device, cfg=cfg)
        print(f"✅ NativeEngineWrapper creado")
        
        # Verificar cuántas partículas se agregaron
        matter_count = wrapper.native_engine.get_matter_count()
        print(f"\n📊 Partículas en motor nativo después de inicialización: {matter_count}")
        print(f"   Células significativas esperadas: {significant_cells}")
        
        if matter_count == 0:
            print(f"\n❌ CRÍTICO: El motor nativo está vacío después de inicialización.")
            print(f"❌ _initialize_native_state_from_dense() no está agregando partículas.")
            return False
        elif matter_count < significant_cells * 0.5:  # Menos del 50% de lo esperado
            print(f"⚠️ ADVERTENCIA: Menos partículas ({matter_count}) que células significativas ({significant_cells})")
            print(f"⚠️ Puede haber un problema con el sampling o el umbral")
        
        # Verificar coordenadas activas
        active_coords = wrapper.native_engine.get_active_coords()
        print(f"\n📊 Coordenadas activas: {len(active_coords)}")
        expected_max = grid_size * grid_size * 2
        if len(active_coords) > expected_max:
            print(f"⚠️ ADVERTENCIA: Demasiadas coordenadas activas ({len(active_coords)} > {expected_max})")
        
        # Intentar recuperar estado denso
        print(f"\n🔍 Intentando convertir de vuelta a estado denso...")
        dense_state = wrapper.get_dense_state()
        
        if dense_state is None:
            print(f"❌ CRÍTICO: get_dense_state() retornó None")
            return False
        
        dense_abs_max = dense_state.abs().max().item()
        print(f"📊 Estado denso recuperado: max abs={dense_abs_max:.6e}")
        
        if dense_abs_max < 1e-10:
            print(f"\n❌ CRÍTICO: El estado denso recuperado está VACÍO (max abs={dense_abs_max:.6e})")
            print(f"❌ Esto significa que get_state_at() está retornando solo vacío cuántico")
            return False
        
        print(f"\n✅ Estado inicial denso se genera y convierte correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de conversión: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_initial_state()
    
    if success:
        print("\n✅ TODAS LAS PRUEBAS PASARON")
        sys.exit(0)
    else:
        print("\n❌ LAS PRUEBAS FALLARON")
        sys.exit(1)

