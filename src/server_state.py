# src/server_state.py
import asyncio
import json
import logging
from .history_manager import SimulationHistory

# Estado global de la aplicación
g_state = {
    "websockets": {},  # {ws_id: ws_object}
    "training_process": None,
    "motor": None,
    "is_paused": True,
    "simulation_step": 0,
    "viz_type": "density",
    "inference_running": False,
    "active_experiment": None,
    "simulation_speed": 1.0,  # Multiplicador de velocidad
    "target_fps": 10.0,  # FPS objetivo
    "frame_skip": 0,  # Frames a saltar (0 = todos, 1 = cada otro, etc.)
    "simulation_history": SimulationHistory(max_frames=1000),  # Historial de simulación
    "history_enabled": False,  # Habilitar/deshabilitar guardado de historia
    "live_feed_enabled": True,  # Habilitar/deshabilitar envío de datos en tiempo real (optimización)
}

# --- INICIO DE LA CORRECCIÓN ---
# Movemos la lógica de comunicación aquí para
# evitar importaciones circulares.

async def broadcast(data):
    """Envía un mensaje JSON a todos los clientes WebSocket conectados."""
    if not g_state['websockets']:
        return
        
    tasks = []
    for ws in list(g_state['websockets'].values()):
        if not ws.closed:
            tasks.append(ws.send_json(data))
    
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
# --- FIN DE LA CORRECCIÓN ---