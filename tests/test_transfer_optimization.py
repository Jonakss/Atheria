#!/usr/bin/env python3
"""
Script de prueba para comparar transferencia JSON vs Binary optimizado.

Ejecuta este script para ver las mejoras de rendimiento y tamaño.
"""
import numpy as np
import json
import time
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.server.data_compression import optimize_frame_payload, get_payload_size
    from src.server.data_transfer_optimized import optimize_frame_payload_binary, encode_frame_binary, decode_frame_binary
    OLD_AVAILABLE = True
    NEW_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    sys.exit(1)


def create_test_frame(grid_size=256):
    """Crea un frame de prueba con datos simulados."""
    # Generar map_data (grid principal)
    map_data = np.random.rand(grid_size, grid_size).astype(np.float32) * 2.0 - 1.0  # [-1, 1]
    
    # Generar complex_3d_data
    complex_real = np.random.rand(grid_size, grid_size).astype(np.float32)
    complex_imag = np.random.rand(grid_size, grid_size).astype(np.float32)
    
    # Generar flow_data
    flow_dx = np.random.rand(grid_size, grid_size).astype(np.float32) * 0.1
    flow_dy = np.random.rand(grid_size, grid_size).astype(np.float32) * 0.1
    flow_magnitude = np.sqrt(flow_dx**2 + flow_dy**2)
    
    return {
        "step": 1000,
        "timestamp": time.time(),
        "map_data": map_data.tolist(),
        "complex_3d_data": {
            "real": complex_real.tolist(),
            "imag": complex_imag.tolist()
        },
        "flow_data": {
            "dx": flow_dx.tolist(),
            "dy": flow_dy.tolist(),
            "magnitude": flow_magnitude.tolist()
        },
        "hist_data": {},
        "poincare_coords": [],
        "simulation_info": {
            "step": 1000,
            "is_paused": False,
            "live_feed_enabled": True
        }
    }


async def benchmark_comparison():
    """Compara rendimiento entre método actual y optimizado."""
    print("🔬 Benchmark: JSON vs Binary Optimizado\n")
    
    # Crear frame de prueba
    test_frame = create_test_frame(grid_size=256)
    print(f"📊 Frame de prueba: Grid 256x256")
    print(f"   - map_data: {256*256*4/1024:.1f} KB (float32)")
    print(f"   - complex_3d_data: {256*256*4*2/1024:.1f} KB")
    print(f"   - flow_data: {256*256*4*3/1024:.1f} KB")
    print(f"   - Total sin compresión: ~{(256*256*4*6)/1024:.1f} KB\n")
    
    # --- Método Actual (JSON + zlib + base64) ---
    print("📦 Método Actual (JSON + zlib + base64):")
    try:
        start = time.time()
        optimized_json = await optimize_frame_payload(
            test_frame,
            enable_compression=True,
            downsample_factor=1,
            viz_type='density'
        )
        json_time = (time.time() - start) * 1000  # ms
        
        # Serializar a JSON string
        json_str = json.dumps(optimized_json, separators=(',', ':'))
        json_size = len(json_str.encode('utf-8'))
        
        print(f"   ✅ Tiempo de optimización: {json_time:.2f} ms")
        print(f"   ✅ Tamaño final: {json_size/1024:.1f} KB")
        print(f"   ✅ Ratio de compresión: {(256*256*4*6)/json_size:.1f}x\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        json_size = 0
        json_time = 0
    
    # --- Método Optimizado (Binary + Quantization + LZ4) ---
    print("⚡ Método Optimizado (Binary + Quantization + LZ4):")
    try:
        start = time.time()
        binary_data = await optimize_frame_payload_binary(
            test_frame,
            use_quantization=True,
            use_differential=False  # Sin differential para comparación justa
        )
        binary_time = (time.time() - start) * 1000  # ms
        
        binary_size = len(binary_data)
        
        # Verificar que se puede decodificar
        decoded = decode_frame_binary(binary_data)
        
        print(f"   ✅ Tiempo de optimización: {binary_time:.2f} ms")
        print(f"   ✅ Tamaño final: {binary_size/1024:.1f} KB")
        print(f"   ✅ Ratio de compresión: {(256*256*4*6)/binary_size:.1f}x")
        print(f"   ✅ Decodificación exitosa: {len(decoded.get('map_data', []))} elementos\n")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        binary_size = 0
        binary_time = 0
    
    # --- Comparación Final ---
    if json_size > 0 and binary_size > 0:
        size_reduction = ((json_size - binary_size) / json_size) * 100
        speed_improvement = json_time / binary_time if binary_time > 0 else 0
        
        print("📊 Resultados:")
        print(f"   🎯 Reducción de tamaño: {size_reduction:.1f}%")
        print(f"   🚀 Mejora de velocidad: {speed_improvement:.1f}x")
        print(f"   💾 Ahorro de ancho de banda: {json_size/binary_size:.1f}x")
        print(f"\n   📈 A 10 FPS:")
        print(f"      - Actual: {json_size*10/1024:.1f} KB/s")
        print(f"      - Optimizado: {binary_size*10/1024:.1f} KB/s")
        print(f"      - Ahorro: {(json_size - binary_size)*10/1024:.1f} KB/s")


if __name__ == "__main__":
    import asyncio
    asyncio.run(benchmark_comparison())

