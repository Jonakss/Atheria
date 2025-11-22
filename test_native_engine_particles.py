#!/usr/bin/env python3
"""
Script de diagnóstico para verificar que el motor nativo almacena y recupera partículas correctamente.
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

def test_native_engine():
    """Prueba básica del motor nativo: agregar y recuperar partículas."""
    try:
        import atheria_core
        print("✅ Módulo atheria_core importado exitosamente")
    except ImportError as e:
        print(f"❌ Error importando atheria_core: {e}")
        return False
    
    print("\n" + "="*80)
    print("PRUEBA 1: Verificar que se pueden agregar y recuperar partículas")
    print("="*80)
    
    # Crear motor nativo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_size = 64
    d_state = 8
    
    print(f"📊 Configuración: grid_size={grid_size}, d_state={d_state}, device={device}")
    
    try:
        engine = atheria_core.Engine(d_state, device, grid_size)
        print(f"✅ Motor nativo creado exitosamente")
    except Exception as e:
        print(f"❌ Error creando motor nativo: {e}")
        return False
    
    # Verificar estado inicial
    initial_matter = engine.get_matter_count()
    print(f"📊 Estado inicial: {initial_matter} partículas almacenadas")
    
    # Agregar algunas partículas de prueba
    num_test_particles = 10
    test_coords = []
    print(f"\n🛠️ Agregando {num_test_particles} partículas de prueba...")
    
    for i in range(num_test_particles):
        x = (i * 7) % grid_size  # Distribuir partículas
        y = (i * 13) % grid_size
        z = 0
        
        # Crear estado de prueba con valores significativos
        test_state = torch.randn(d_state, dtype=torch.complex64, device=device) * 0.5
        test_state_abs_max = test_state.abs().max().item()
        
        coord = atheria_core.Coord3D(x, y, z)
        engine.add_particle(coord, test_state)
        test_coords.append((coord, test_state_abs_max))
        
        print(f"  ✅ Agregada partícula {i+1}/{num_test_particles} en ({x}, {y}): abs_max={test_state_abs_max:.6e}")
    
    # Verificar matter_count después de agregar
    final_matter = engine.get_matter_count()
    print(f"\n📊 Después de agregar: {final_matter} partículas almacenadas (esperado: {num_test_particles})")
    
    if final_matter != num_test_particles:
        print(f"⚠️ ADVERTENCIA: matter_count ({final_matter}) no coincide con partículas agregadas ({num_test_particles})")
        if final_matter == 0:
            print(f"❌ CRÍTICO: matter_map_ está vacío. add_particle() NO está funcionando.")
            return False
    
    # Intentar recuperar las partículas
    print(f"\n🔍 Intentando recuperar las {num_test_particles} partículas agregadas...")
    recovered_count = 0
    empty_count = 0
    none_count = 0
    
    for i, (coord, expected_abs) in enumerate(test_coords):
        retrieved_state = engine.get_state_at(coord)
        
        if retrieved_state is None:
            none_count += 1
            print(f"  ❌ Partícula {i+1} en ({coord.x}, {coord.y}): get_state_at() retornó None")
        else:
            retrieved_abs = retrieved_state.abs().max().item()
            if retrieved_abs > 1e-10:
                recovered_count += 1
                print(f"  ✅ Partícula {i+1} en ({coord.x}, {coord.y}): recuperada (abs_max={retrieved_abs:.6e}, esperado ~{expected_abs:.6e})")
            else:
                empty_count += 1
                print(f"  ⚠️ Partícula {i+1} en ({coord.x}, {coord.y}): estado vacío (abs_max={retrieved_abs:.6e}, esperado ~{expected_abs:.6e})")
    
    print(f"\n📊 RESUMEN DE RECUPERACIÓN:")
    print(f"  ✅ Recuperadas correctamente: {recovered_count}/{num_test_particles}")
    print(f"  ⚠️ Vacías (solo vacío cuántico): {empty_count}/{num_test_particles}")
    print(f"  ❌ None retornado: {none_count}/{num_test_particles}")
    
    if recovered_count == 0:
        print(f"\n❌ CRÍTICO: Ninguna partícula fue recuperable.")
        print(f"❌ PROBLEMA: get_state_at() está retornando solo vacío cuántico.")
        return False
    elif recovered_count < num_test_particles:
        print(f"\n⚠️ ADVERTENCIA: Solo {recovered_count}/{num_test_particles} partículas fueron recuperables.")
        print(f"⚠️ Puede haber un problema con el almacenamiento o recuperación.")
    
    # Probar get_active_coords
    print(f"\n" + "="*80)
    print("PRUEBA 2: Verificar get_active_coords()")
    print("="*80)
    
    try:
        active_coords = engine.get_active_coords()
        print(f"📊 get_active_coords() retornó {len(active_coords)} coordenadas activas")
        
        expected_max = grid_size * grid_size * 2  # Permitir hasta 2x el grid
        if len(active_coords) > expected_max:
            print(f"⚠️ ADVERTENCIA: Demasiadas coordenadas activas ({len(active_coords)} > {expected_max})")
            print(f"⚠️ Puede haber duplicados o un bug en get_active_coords()")
        else:
            print(f"✅ Número de coordenadas activas es razonable ({len(active_coords)} <= {expected_max})")
        
        # Verificar una muestra de coordenadas activas
        print(f"\n🔍 Verificando muestra de {min(5, len(active_coords))} coordenadas activas...")
        for i, coord in enumerate(active_coords[:5]):
            state = engine.get_state_at(coord)
            if state is not None:
                abs_max = state.abs().max().item()
                if abs_max > 1e-10:
                    print(f"  ✅ Coord ({coord.x}, {coord.y}): tiene materia (abs_max={abs_max:.6e})")
                else:
                    print(f"  ⚠️ Coord ({coord.x}, {coord.y}): vacía (abs_max={abs_max:.6e})")
            else:
                print(f"  ❌ Coord ({coord.x}, {coord.y}): retornó None")
    
    except Exception as e:
        print(f"❌ Error obteniendo coordenadas activas: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*80)
    print("PRUEBA 3: Verificar que las coordenadas agregadas están en active_coords")
    print("="*80)
    
    try:
        active_coords = engine.get_active_coords()
        active_coords_set = {(c.x, c.y, c.z) for c in active_coords}
        
        found_in_active = 0
        for coord, _ in test_coords:
            coord_key = (coord.x, coord.y, coord.z)
            if coord_key in active_coords_set:
                found_in_active += 1
                print(f"  ✅ Coord ({coord.x}, {coord.y}) está en active_coords")
            else:
                print(f"  ⚠️ Coord ({coord.x}, {coord.y}) NO está en active_coords")
        
        print(f"\n📊 RESUMEN: {found_in_active}/{num_test_particles} coordenadas agregadas están en active_coords")
        
        if found_in_active == 0:
            print(f"❌ CRÍTICO: Ninguna coordenada agregada está en active_coords.")
            print(f"❌ activate_neighborhood() NO está funcionando correctamente.")
    
    except Exception as e:
        print(f"❌ Error verificando coordenadas: {e}")
    
    print(f"\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    
    if recovered_count == num_test_particles and final_matter == num_test_particles:
        print("✅ TODAS LAS PRUEBAS PASARON: El motor nativo funciona correctamente")
        return True
    elif recovered_count > 0:
        print("⚠️ PRUEBAS PARCIALMENTE EXITOSAS: Algunas partículas se recuperan")
        print("⚠️ Hay un problema parcial con el almacenamiento o recuperación")
        return False
    else:
        print("❌ PRUEBAS FALLARON: El motor nativo NO está funcionando correctamente")
        return False


if __name__ == "__main__":
    print("🧪 DIAGNÓSTICO DEL MOTOR NATIVO")
    print("="*80)
    print("Este script verifica si el motor nativo puede almacenar y recuperar partículas correctamente.\n")
    
    success = test_native_engine()
    
    if success:
        print("\n✅ El motor nativo está funcionando correctamente.")
        sys.exit(0)
    else:
        print("\n❌ El motor nativo tiene problemas. Revisa los logs arriba.")
        sys.exit(1)

