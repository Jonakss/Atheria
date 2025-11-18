# src/data_compression.py
"""
Módulo de compresión y optimización de datos para transferencia WebSocket.
Reduce el tamaño de los datos enviados usando compresión y optimizaciones específicas.
"""
import json
import zlib
import base64
import numpy as np
from typing import Any, Dict, Optional
import logging
import asyncio

def _compress_array_sync(arr: np.ndarray, dtype: str = 'float32') -> Dict[str, Any]:
    """
    Comprime un array NumPy a formato binario comprimido.
    
    Args:
        arr: Array NumPy a comprimir
        dtype: Tipo de dato a usar (default: 'float32' para reducir tamaño)
    
    Returns:
        Dict con 'data' (base64), 'shape', 'dtype', 'compressed': True
    """
    # Convertir a dtype más pequeño si es necesario
    if dtype == 'float32' and arr.dtype == 'float64':
        arr = arr.astype(np.float32)
    
    # Serializar a bytes
    arr_bytes = arr.tobytes()
    
    # Comprimir con zlib (rápido y eficiente)
    compressed = zlib.compress(arr_bytes, level=6)  # Balance entre velocidad y compresión
    
    # Codificar a base64 para JSON
    data_b64 = base64.b64encode(compressed).decode('ascii')
    
    return {
        'data': data_b64,
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'compressed': True
    }

async def compress_array(arr: np.ndarray, dtype: str = 'float32') -> Dict[str, Any]:
    """
    Versión asíncrona de compress_array para no bloquear el event loop.
    Ejecuta la compresión en un thread separado.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compress_array_sync, arr, dtype)

def decompress_array(compressed_data: Dict[str, Any]) -> np.ndarray:
    """
    Descomprime un array comprimido.
    
    Args:
        compressed_data: Dict con 'data', 'shape', 'dtype'
    
    Returns:
        Array NumPy descomprimido
    """
    # Decodificar base64
    compressed_bytes = base64.b64decode(compressed_data['data'])
    
    # Descomprimir
    arr_bytes = zlib.decompress(compressed_bytes)
    
    # Convertir de bytes a array
    dtype_str = compressed_data['dtype']
    # Convertir string de dtype a numpy dtype
    if 'float32' in dtype_str:
        np_dtype = np.float32
    elif 'float64' in dtype_str:
        np_dtype = np.float64
    else:
        np_dtype = np.float32  # Default
    
    arr = np.frombuffer(arr_bytes, dtype=np_dtype)
    
    # Reshape
    arr = arr.reshape(compressed_data['shape'])
    
    return arr

async def optimize_frame_payload(payload: Dict[str, Any], enable_compression: bool = True, 
                          downsample_factor: int = 1, viz_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Optimiza el payload de un frame antes de enviarlo.
    
    Args:
        payload: Payload original con map_data, etc.
        enable_compression: Si True, comprimir arrays grandes
        downsample_factor: Factor de downsampling (1 = sin downsampling)
        viz_type: Tipo de visualización para optimizar selectivamente
    
    Returns:
        Payload optimizado (posiblemente modificado)
    """
    optimized = payload.copy()
    
    # Optimizar map_data (siempre se envía)
    if 'map_data' in payload and payload['map_data']:
        map_data = payload['map_data']
        
        # Si es lista de listas, convertir a numpy
        if isinstance(map_data, list):
            try:
                map_array = np.array(map_data, dtype=np.float32)
                
                # Aplicar downsampling si se especifica
                if downsample_factor > 1 and map_array.size > 0:
                    H, W = map_array.shape[0], map_array.shape[1]
                    new_H, new_W = H // downsample_factor, W // downsample_factor
                    if new_H > 0 and new_W > 0:
                        # Downsample usando promedio
                        map_array = map_array[:new_H * downsample_factor, :new_W * downsample_factor]
                        map_array = map_array.reshape(new_H, downsample_factor, new_W, downsample_factor).mean(axis=(1, 3))
                
                # Comprimir si está habilitado y el array es grande
                # Usar threshold más alto (50k elementos) para evitar overhead en arrays pequeños
                if enable_compression and map_array.size > 50000:  # Solo comprimir si > 50k elementos
                    optimized['map_data'] = await compress_array(map_array)
                else:
                    optimized['map_data'] = map_array.tolist()
                    
            except Exception as e:
                logging.warning(f"Error optimizando map_data: {e}")
                # Fallback a original
                optimized['map_data'] = map_data
    
    # Optimizar complex_3d_data solo si está presente y se necesita
    if 'complex_3d_data' in payload and payload['complex_3d_data']:
        if viz_type == 'complex_3d':  # Solo optimizar si se está usando
            try:
                complex_data = payload['complex_3d_data']
                
                # Comprimir real e imag por separado
                if 'real' in complex_data and isinstance(complex_data['real'], list):
                    real_array = np.array(complex_data['real'], dtype=np.float32)
                    if enable_compression and real_array.size > 50000:
                        optimized['complex_3d_data']['real'] = await compress_array(real_array)
                
                if 'imag' in complex_data and isinstance(complex_data['imag'], list):
                    imag_array = np.array(complex_data['imag'], dtype=np.float32)
                    if enable_compression and imag_array.size > 50000:
                        optimized['complex_3d_data']['imag'] = await compress_array(imag_array)
                        
            except Exception as e:
                logging.warning(f"Error optimizando complex_3d_data: {e}")
    
    # Optimizar flow_data solo si está presente y se necesita
    if 'flow_data' in payload and payload['flow_data'] and viz_type == 'flow':
        try:
            flow_data = payload['flow_data']
            
            # Comprimir dx, dy, magnitude si son grandes
            for key in ['dx', 'dy', 'magnitude']:
                if key in flow_data and isinstance(flow_data[key], list):
                    flow_array = np.array(flow_data[key], dtype=np.float32)
                    if enable_compression and flow_array.size > 50000:
                        optimized['flow_data'][key] = await compress_array(flow_array)
                        
        except Exception as e:
            logging.warning(f"Error optimizando flow_data: {e}")
    
    # Optimizar phase_hsv_data solo si está presente y se necesita
    if 'phase_hsv_data' in payload and payload['phase_hsv_data'] and viz_type == 'phase_hsv':
        try:
            hsv_data = payload['phase_hsv_data']
            for key in ['hue', 'saturation', 'value']:
                if key in hsv_data and isinstance(hsv_data[key], list):
                    hsv_array = np.array(hsv_data[key], dtype=np.float32)
                    if enable_compression and hsv_array.size > 50000:
                        optimized['phase_hsv_data'][key] = await compress_array(hsv_array)
                        
        except Exception as e:
            logging.warning(f"Error optimizando phase_hsv_data: {e}")
    
    # No optimizar poincare_coords (pequeño), hist_data (ya pequeño), phase_attractor (ya pequeño)
    
    return optimized

def get_payload_size(payload: Dict[str, Any]) -> int:
    """
    Calcula el tamaño aproximado del payload en bytes.
    
    Args:
        payload: Payload a medir
    
    Returns:
        Tamaño en bytes (aproximado)
    """
    try:
        # Serializar a JSON string para medir
        json_str = json.dumps(payload, separators=(',', ':'))
        return len(json_str.encode('utf-8'))
    except Exception:
        return 0

