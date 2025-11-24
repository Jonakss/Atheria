"""
Módulo de serialización eficiente para separar datos de visualización (binario)
de datos del servidor (JSON).

Estrategia:
- JSON: Solo para comandos, notificaciones, metadatos del servidor (pequeños)
- Binario (CBOR/MessagePack): Para frames de visualización (grandes, arrays numéricos)
"""
import json
import logging
from typing import Any, Dict, Optional, Tuple

# Intentar importar librerías de serialización binaria
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    logging.debug("msgpack no disponible. Usando CBOR o fallback.")

try:
    import cbor2
    CBOR_AVAILABLE = True
except ImportError:
    CBOR_AVAILABLE = False
    logging.debug("cbor2 no disponible. Usando JSON como fallback.")

def serialize_frame_binary(payload: Dict[str, Any]) -> Tuple[bytes, str]:
    """
    Serializa un frame de visualización a formato binario eficiente.
    
    Args:
        payload: Payload del frame con map_data, hist_data, etc.
    
    Returns:
        Tuple: (bytes binarios, formato usado: "msgpack", "cbor", o "json")
    """
    # Convertir listas a arrays para mejor compresión
    optimized_payload = payload.copy()
    
    # Optimizar map_data si existe (suele ser el más grande)
    if 'map_data' in optimized_payload:
        map_data = optimized_payload['map_data']
        # Si es numpy array, convertir a lista para evitar error de serialización
        # Esto es un fallback por si data_compression falló o no se usó
        if hasattr(map_data, 'tolist'):
            optimized_payload['map_data'] = map_data.tolist()
    
    # Intentar usar MessagePack primero (más eficiente para arrays numéricos)
    if MSGPACK_AVAILABLE:
        try:
            # MessagePack serializa arrays numéricos muy eficientemente
            binary_data = msgpack.packb(optimized_payload, use_bin_type=True)
            return binary_data, "msgpack"
        except Exception as e:
            logging.warning(f"Error serializando con msgpack: {e}, usando fallback")
    
    # Fallback a CBOR (bueno para arrays binarios)
    if CBOR_AVAILABLE:
        try:
            binary_data = cbor2.dumps(optimized_payload)
            return binary_data, "cbor"
        except Exception as e:
            logging.warning(f"Error serializando con CBOR: {e}, usando JSON")
    
    # Fallback a JSON (último recurso)
    json_str = json.dumps(optimized_payload, separators=(',', ':'))
    return json_str.encode('utf-8'), "json"


def deserialize_frame_binary(data: bytes, format_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Deserializa un frame binario a diccionario Python.
    
    Args:
        data: Bytes binarios a deserializar
        format_hint: Formato esperado ("msgpack", "cbor", "json", o None para auto-detectar)
    
    Returns:
        Dict con el payload deserializado
    """
    # Auto-detectar formato si no se proporciona hint
    if format_hint is None:
        # Intentar detectar formato por el primer byte
        if len(data) > 0:
            first_byte = data[0]
            # MessagePack: rango 0x80-0xFF para objetos, arrays, etc.
            if first_byte >= 0x80 and first_byte <= 0xFF:
                format_hint = "msgpack"
            # CBOR: típicamente 0xA0-0xBF para mapas, 0x80-0x9F para arrays
            elif first_byte >= 0xA0 and first_byte <= 0xBF:
                format_hint = "cbor"
            # JSON: típicamente '{' (0x7B) o '[' (0x5B)
            elif first_byte in (0x7B, 0x5B):
                format_hint = "json"
            else:
                # Default: intentar MessagePack primero
                format_hint = "msgpack"
    
    # Deserializar según formato
    if format_hint == "msgpack" and MSGPACK_AVAILABLE:
        try:
            return msgpack.unpackb(data, raw=False)
        except Exception as e:
            logging.warning(f"Error deserializando msgpack: {e}, intentando CBOR")
            format_hint = "cbor"
    
    if format_hint == "cbor" and CBOR_AVAILABLE:
        try:
            return cbor2.loads(data)
        except Exception as e:
            logging.warning(f"Error deserializando CBOR: {e}, usando JSON")
            format_hint = "json"
    
    # Fallback a JSON
    try:
        json_str = data.decode('utf-8')
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Error deserializando datos: {e}. Formato: {format_hint}")


def should_use_binary(message_type: str, payload: Optional[Dict[str, Any]] = None) -> bool:
    """
    Determina si un mensaje debe enviarse como binario o JSON.
    
    Reglas:
    - simulation_frame: Binario (datos grandes de visualización)
    - Comandos, notificaciones, estado del servidor: JSON (pequeños)
    
    Args:
        message_type: Tipo de mensaje ("simulation_frame", "notification", etc.)
        payload: Payload opcional para análisis adicional
    
    Returns:
        True si debe usar binario, False si debe usar JSON
    """
    # Frames de visualización siempre binario
    if message_type == "simulation_frame":
        return True
    
    # Todo lo demás es JSON
    return False

