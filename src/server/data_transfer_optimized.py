"""
Optimización avanzada de transferencia de datos para WebSocket.

Mejoras sobre data_compression.py:
1. Binary WebSocket frames (sin base64 overhead)
2. LZ4 compression (más rápido que zlib)
3. Differential compression (solo cambios)
4. Quantization (float32→uint8 para visualización)
5. CBOR para metadata (más eficiente que JSON)
"""
import numpy as np
import struct
from typing import Any, Dict, Optional, Tuple, BinaryIO
import logging
import asyncio

# Intentar importar librerías opcionales
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logging.warning("lz4 no disponible. Usando zlib como fallback.")

try:
    import cbor2
    CBOR_AVAILABLE = True
except ImportError:
    CBOR_AVAILABLE = False
    logging.warning("cbor2 no disponible. Usando JSON como fallback.")

try:
    import zlib
    ZLIB_AVAILABLE = True
except ImportError:
    ZLIB_AVAILABLE = False
    logging.warning("zlib no disponible. Sin compresión disponible.")


def _compress_bytes_lz4(data_bytes: bytes) -> bytes:
    """
    Comprime bytes con LZ4 (muy rápido, buena compresión).
    
    Args:
        data_bytes: bytes a comprimir
    
    Returns:
        bytes comprimidos
    """
    # Comprimir con LZ4 (muy rápido, buena compresión para datos numéricos)
    if LZ4_AVAILABLE:
        compressed = lz4.frame.compress(data_bytes)
    elif ZLIB_AVAILABLE:
        compressed = zlib.compress(data_bytes, level=1)  # Nivel bajo para velocidad
    else:
        compressed = data_bytes  # Sin compresión si no hay librerías
    
    return compressed


def _decompress_bytes_lz4(compressed_bytes: bytes) -> bytes:
    """
    Descomprime bytes comprimidos con LZ4 o zlib.
    
    Args:
        compressed_bytes: bytes comprimidos
    
    Returns:
        bytes descomprimidos
    """
    if LZ4_AVAILABLE:
        try:
            return lz4.frame.decompress(compressed_bytes)
        except:
            # Fallback a zlib si falla
            if ZLIB_AVAILABLE:
                return zlib.decompress(compressed_bytes)
            return compressed_bytes
    elif ZLIB_AVAILABLE:
        return zlib.decompress(compressed_bytes)
    else:
        return compressed_bytes


def _quantize_to_uint8(arr: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[bytes, float, float]:
    """
    Cuantiza un array float32 a uint8 para reducir tamaño.
    
    Args:
        arr: Array NumPy (float32 o float64)
        min_val: Valor mínimo (auto-detectado si None)
        max_val: Valor máximo (auto-detectado si None)
    
    Returns:
        Tuple: (bytes uint8, min_val, max_val) para desquantizar
    """
    arr_float32 = arr.astype(np.float32)
    
    if min_val is None:
        min_val = float(np.min(arr_float32))
    if max_val is None:
        max_val = float(np.max(arr_float32))
    
    # Normalizar a [0, 1]
    if max_val != min_val:
        normalized = (arr_float32 - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(arr_float32)
    
    # Cuantizar a uint8
    quantized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    
    return quantized.tobytes(), min_val, max_val


def _dequantize_from_uint8(data: bytes, shape: Tuple[int, ...], min_val: float, max_val: float) -> np.ndarray:
    """
    Descuantiza un array uint8 a float32.
    
    Args:
        data: bytes uint8
        shape: Shape del array original
        min_val: Valor mínimo original
        max_val: Valor máximo original
    
    Returns:
        Array NumPy float32 desquantizado
    """
    quantized = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    
    # Desnormalizar de [0, 1] a [min_val, max_val]
    normalized = quantized.astype(np.float32) / 255.0
    
    if max_val != min_val:
        arr = normalized * (max_val - min_val) + min_val
    else:
        arr = np.full_like(normalized, min_val, dtype=np.float32)
    
    return arr


def encode_array_binary(arr: np.ndarray, use_quantization: bool = True, compress: bool = True) -> Dict[str, Any]:
    """
    Codifica un array NumPy a formato binario optimizado.
    
    Args:
        arr: Array NumPy a codificar
        use_quantization: Si True, cuantizar a uint8 (reduce tamaño 4x)
        compress: Si True, comprimir con LZ4
    
    Returns:
        Dict con datos binarios listos para enviar
    """
    shape = arr.shape
    dtype_str = str(arr.dtype)
    
    if use_quantization and arr.dtype in [np.float32, np.float64]:
        # Cuantizar a uint8 (reducción de 4x en tamaño)
        data_bytes, min_val, max_val = _quantize_to_uint8(arr)
        quantized = True
        original_dtype = dtype_str
    else:
        # Mantener dtype original
        data_bytes = arr.tobytes()
        min_val = max_val = None
        quantized = False
        original_dtype = None
    
    # Comprimir si está habilitado y el array es grande
    if compress and len(data_bytes) > 4096:  # Solo comprimir arrays > 4KB
        compressed_bytes = _compress_bytes_lz4(data_bytes)
        is_compressed = True
        data_bytes = compressed_bytes
    else:
        is_compressed = False
    
    return {
        'data': data_bytes,  # bytes raw (sin base64)
        'shape': shape,
        'dtype': 'uint8' if quantized else dtype_str,
        'original_dtype': original_dtype,
        'quantized': quantized,
        'compressed': is_compressed,
        'min_val': min_val,
        'max_val': max_val,
        'format': 'binary'  # Indica que es formato binario (no base64)
    }


def decode_array_binary(encoded: Dict[str, Any]) -> np.ndarray:
    """
    Decodifica un array binario codificado.
    
    Args:
        encoded: Dict con datos binarios codificados
    
    Returns:
        Array NumPy descomprimido y desquantizado
    """
    data_bytes = encoded['data']  # bytes raw (ya no base64)
    shape = tuple(encoded['shape'])
    dtype_str = encoded['dtype']
    is_compressed = encoded.get('compressed', False)
    is_quantized = encoded.get('quantized', False)
    
    # Descomprimir si es necesario
    if is_compressed:
        data_bytes = _decompress_bytes_lz4(data_bytes)
    
    # Descuantizar si es necesario
    if is_quantized:
        min_val = encoded['min_val']
        max_val = encoded['max_val']
        arr = _dequantize_from_uint8(data_bytes, shape, min_val, max_val)
    else:
        # Convertir dtype string a numpy dtype
        np_dtype = np.dtype(dtype_str)
        arr = np.frombuffer(data_bytes, dtype=np_dtype).reshape(shape)
    
    return arr


def encode_frame_binary(payload: Dict[str, Any], use_quantization: bool = True, 
                       use_differential: bool = False, previous_frame: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Codifica un frame completo a formato binario optimizado.
    
    Args:
        payload: Payload del frame (dict con map_data, etc.)
        use_quantization: Si True, cuantizar arrays float
        use_differential: Si True, solo enviar diferencias desde previous_frame
        previous_frame: Frame anterior para differential compression
    
    Returns:
        bytes binarios listos para enviar por WebSocket
    """
    # Convertir arrays grandes a formato binario
    optimized = {}
    
    # Metadata (pequeña, puede usar CBOR o JSON)
    metadata = {
        'step': payload.get('step'),
        'timestamp': payload.get('timestamp'),
        'simulation_info': payload.get('simulation_info', {})
    }
    
    # Codificar map_data (siempre presente y grande)
    if 'map_data' in payload and payload['map_data'] is not None:
        map_data = payload['map_data']
        
        if isinstance(map_data, list):
            map_array = np.array(map_data, dtype=np.float32)
        elif isinstance(map_data, np.ndarray):
            map_array = map_data.astype(np.float32)
        else:
            map_array = map_data
        
        # Differential compression opcional
        if use_differential and previous_frame and 'map_data' in previous_frame:
            try:
                prev_map = np.array(previous_frame['map_data'], dtype=np.float32)
                if prev_map.shape == map_array.shape:
                    # Enviar solo diferencias
                    diff = map_array - prev_map
                    # Si los cambios son pequeños, usar differential
                    if np.abs(diff).max() < 0.1:  # Threshold arbitrario
                        encoded_map = encode_array_binary(diff, use_quantization=True, compress=True)
                        encoded_map['is_differential'] = True
                        optimized['map_data'] = encoded_map
                    else:
                        # Cambios grandes, enviar completo
                        optimized['map_data'] = encode_array_binary(map_array, use_quantization, compress=True)
                else:
                    optimized['map_data'] = encode_array_binary(map_array, use_quantization, compress=True)
            except Exception as e:
                logging.debug(f"Error en differential compression: {e}")
                optimized['map_data'] = encode_array_binary(map_array, use_quantization, compress=True)
        else:
            optimized['map_data'] = encode_array_binary(map_array, use_quantization, compress=True)
    
    # Codificar otros arrays grandes solo si están presentes
    for key in ['complex_3d_data', 'flow_data', 'phase_hsv_data']:
        if key in payload and payload[key] is not None:
            data = payload[key]
            if isinstance(data, dict):
                encoded_sub = {}
                for subkey, subval in data.items():
                    if isinstance(subval, (list, np.ndarray)):
                        subarr = np.array(subval, dtype=np.float32) if isinstance(subval, list) else subval.astype(np.float32)
                        encoded_sub[subkey] = encode_array_binary(subarr, use_quantization, compress=True)
                    else:
                        encoded_sub[subkey] = subval
                optimized[key] = encoded_sub
            else:
                optimized[key] = data
    
    # Datos pequeños sin optimizar
    for key in ['hist_data', 'poincare_coords', 'phase_attractor', 'roi_info']:
        if key in payload:
            optimized[key] = payload[key]
    
    # Serializar metadata y arrays a formato binario
    # Usar CBOR si está disponible (más eficiente que JSON)
    if CBOR_AVAILABLE:
        # Para arrays binarios, CBOR los maneja bien
        # Necesitamos una estructura especial para los bytes
        frame_data = {
            'metadata': metadata,
            'arrays': {}
        }
        
        # Separar metadata de arrays binarios
        for key, value in optimized.items():
            if isinstance(value, dict) and value.get('format') == 'binary':
                # Array binario
                frame_data['arrays'][key] = value
            else:
                # Metadata o datos pequeños
                if 'metadata' not in frame_data:
                    frame_data['metadata'] = {}
                frame_data['metadata'][key] = value
        
        return cbor2.dumps(frame_data)
    else:
        # Fallback: JSON (con arrays binarios como estructura especial)
        import json
        # Convertir bytes a estructura serializable para JSON
        frame_json = {
            'metadata': metadata,
            'arrays': {}
        }
        
        for key, value in optimized.items():
            if isinstance(value, dict) and value.get('format') == 'binary':
                # Para JSON, necesitamos base64 (pero es menos eficiente)
                import base64
                value_copy = value.copy()
                value_copy['data'] = base64.b64encode(value['data']).decode('ascii')
                frame_json['arrays'][key] = value_copy
            else:
                frame_json['metadata'][key] = value
        
        return json.dumps(frame_json, separators=(',', ':')).encode('utf-8')


def decode_frame_binary(data: bytes) -> Dict[str, Any]:
    """
    Decodifica un frame binario recibido.
    
    Args:
        data: bytes binarios del frame
    
    Returns:
        Dict con datos decodificados (similar al payload original)
    """
    # Detectar formato (CBOR o JSON)
    # Intentar primero CBOR (más común en formato binario)
    frame_data = None
    
    if CBOR_AVAILABLE:
        try:
            frame_data = cbor2.loads(data)
        except Exception:
            # No es CBOR, intentar JSON
            pass
    
    # Si CBOR falló o no está disponible, intentar JSON
    if frame_data is None:
        try:
            import json
            # Intentar decodificar como UTF-8 solo si parece texto
            # JSON válido comienza con '{' o '[' en ASCII
            if data[0] in (0x7b, 0x5b):  # '{' o '['
                frame_data = json.loads(data.decode('utf-8'))
            else:
                # Datos binarios, no JSON válido
                logging.error(f"Error decodificando frame: Datos no son CBOR ni JSON válido. Primer byte: 0x{data[0]:02x}")
                return {}
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logging.error(f"Error decodificando frame como JSON: {e}")
            return {}
        except Exception as e:
            logging.error(f"Error decodificando frame: {e}")
            return {}
    
    # Reconstruir payload
    payload = frame_data.get('metadata', {}).copy()
    arrays = frame_data.get('arrays', {})
    
    # Decodificar arrays binarios
    for key, encoded in arrays.items():
        if encoded.get('format') == 'binary':
            # Si viene de JSON, puede estar en base64
            if isinstance(encoded['data'], str):
                import base64
                encoded['data'] = base64.b64decode(encoded['data'])
            
            arr = decode_array_binary(encoded)
            
            # Si es differential, necesitamos el frame anterior para reconstruir
            if encoded.get('is_differential'):
                # Por ahora, solo decodificar el diff (el frontend manejará la reconstrucción)
                payload[key] = arr
                payload[f'{key}_is_differential'] = True
            else:
                payload[key] = arr.tolist() if arr.size < 1000000 else arr  # Convertir a lista si es pequeño
        else:
            payload[key] = encoded
    
    return payload


# Funciones de ayuda para integración con código existente
async def optimize_frame_payload_binary(payload: Dict[str, Any], 
                                       use_quantization: bool = True,
                                       use_differential: bool = False,
                                       previous_frame: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Versión optimizada binaria de optimize_frame_payload.
    
    Retorna bytes listos para enviar por WebSocket (no JSON string).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        encode_frame_binary, 
        payload, 
        use_quantization, 
        use_differential, 
        previous_frame
    )

