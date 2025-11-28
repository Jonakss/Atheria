# src/server/server_state.py
import asyncio
import json
import logging
from ..managers.history_manager import SimulationHistory
from .data_compression import optimize_frame_payload, get_payload_size
from ..managers.roi_manager import ROIManager, apply_roi_to_payload
from .data_serialization import serialize_frame_binary, should_use_binary

from src.cache import cache

# Estado global de la aplicación
g_state = {
    "websockets": {},  # {ws_id: ws_object}
    "training_process": None,
    "motor": None,
    "cache": cache,  # Caché distribuido (Dragonfly)
    "is_paused": True,
    "simulation_step": 0,
    "viz_type": "density",
    "inference_running": False,
    "active_experiment": None,
    "simulation_speed": 1.0,  # Multiplicador de velocidad
    "target_fps": 10.0,  # FPS objetivo
    "frame_skip": 0,  # Frames a saltar (0 = todos, 1 = cada otro, etc.)
    "simulation_history": SimulationHistory(max_frames=1000),  # Historial de simulación (1000 frames)
    "history_enabled": False,  # Habilitar/deshabilitar guardado de historia
    "history_save_interval": 10,  # Guardar cada N frames en el historial (reducción de memoria)
    "live_feed_enabled": True,  # Habilitar/deshabilitar envío de datos en tiempo real (optimización)
    "data_compression_enabled": True,  # Habilitar compresión de datos WebSocket
    "downsample_factor": 1,  # Factor de downsampling para transferencia (1 = sin downsampling)
    "roi_manager": ROIManager(grid_size=256),  # Gestor de Region of Interest (ROI)
    "analysis_status": "idle",  # Estado del análisis: 'idle', 'running', 'paused'
    "analysis_type": None,  # Tipo de análisis: 'universe_atlas', 'cell_chemistry', None
    "analysis_task": None,  # Tarea de análisis actual (para cancelación)
    "analysis_cancel_event": None,  # Evento para cancelar análisis
    "shutdown_event": None,  # Evento de asyncio para señalizar shutdown del servidor
}

# --- INICIO DE LA CORRECCIÓN ---
# Movemos la lógica de comunicación aquí para
# evitar importaciones circulares.

async def broadcast(data):
    """
    Envía un mensaje a todos los clientes WebSocket conectados.
    
    Estrategia:
    - simulation_frame: Binario (MessagePack/CBOR) para eficiencia
    - Otros mensajes: JSON (comandos, notificaciones, metadatos del servidor)
    """
    if not g_state['websockets']:
        return
    
    message_type = data.get("type", "")
    payload = data.get("payload", {})
    
    # Determinar si debe usar binario o JSON
    use_binary = should_use_binary(message_type, payload)
    
    tasks = []
    for ws in list(g_state['websockets'].values()):
        if ws.closed:
            continue
        
        try:
            if use_binary:
                # Serializar frame de visualización a binario
                binary_data, format_used = serialize_frame_binary(payload)
                
                # Enviar metadata JSON primero (pequeño, ~100 bytes)
                metadata = {
                    "type": f"{message_type}_binary",
                    "format": format_used,
                    "size": len(binary_data)
                }
                
                # Enviar metadata JSON y luego datos binarios
                tasks.append(ws.send_json(metadata))
                tasks.append(ws.send_bytes(binary_data))
                
                logging.debug(f"Enviado frame binario: {message_type}, formato={format_used}, tamaño={len(binary_data)} bytes")
            else:
                # Enviar como JSON (comandos, notificaciones, etc.)
                tasks.append(ws.send_json(data))
                
        except Exception as e:
            logging.warning(f"Error enviando mensaje {message_type}: {e}")
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

async def send_notification(websocket, message: str, status: str = "info"):
    """Envía una notificación a un websocket específico."""
    if websocket and not websocket.closed:
        try:
            await websocket.send_json({
                "type": "notification",
                "payload": {"status": status, "message": message}
            })
        except Exception as e:
            logging.warning(f"Error al enviar notificación: {e}")

async def send_to_websocket(websocket, data_type: str, payload: dict):
    """Envía un mensaje con tipo y payload a un websocket específico."""
    if websocket and not websocket.closed:
        try:
            await websocket.send_json({"type": data_type, "payload": payload})
        except Exception as e:
            logging.warning(f"Error al enviar datos de tipo {data_type}: {e}")

async def broadcast_binary(data: bytes, frame_type: str = "simulation_frame"):
    """
    Envía datos binarios a todos los clientes WebSocket conectados.
    
    Para uso con frames optimizados (data_transfer_optimized).
    Los frames binarios son 5-10x más pequeños que JSON.
    
    Args:
        data: bytes binarios a enviar (CBOR o formato optimizado)
        frame_type: Tipo de frame para identificar en el frontend (default: "simulation_frame")
    
    Nota: aiohttp WebSocketResponse tiene send_bytes() que envía frames binarios.
    Enviamos primero un mensaje JSON pequeño con metadata, luego los datos binarios.
    """
    if not g_state['websockets']:
        return
        
    tasks = []
    for ws in list(g_state['websockets'].values()):
        if not ws.closed:
            try:
                # Estrategia híbrida: Metadata JSON (pequeño) + Datos binarios (grandes)
                # Esto permite al frontend saber qué tipo de frame esperar
                metadata = {
                    "type": f"{frame_type}_binary",
                    "size": len(data),
                    "format": "cbor"  # o "binary" dependiendo del formato
                }
                
                # Enviar metadata JSON primero (pequeño, ~50 bytes)
                tasks.append(ws.send_json(metadata))
                
                # Enviar datos binarios después (grandes, 10-20 KB optimizado)
                # aiohttp WebSocketResponse.send_bytes() envía frames binarios directamente
                tasks.append(ws.send_bytes(data))
                
            except Exception as e:
                logging.warning(f"Error enviando binary frame: {e}")
                # Fallback silencioso: Si falla, no hacer nada (evitar spam de logs)
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
# --- FIN DE LA CORRECCIÓN ---