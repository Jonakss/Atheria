# src/server_state.py
import asyncio
import json

# Estado global de la aplicación
g_state = {
    "websockets": [],
    "training": {
        "process": None,
        "status": "idle", # idle, running, finished, error
        "current_experiment": None,
    },
    "simulation": {
        "is_running": False,
        "current_experiment": None,
        "current_step": 0,
        "pipeline": None,
        "fps": 10,
    }
}

# --- ¡¡FUNCIÓN MOVIDA AQUÍ!! ---
# Se mueve aquí para romper la dependencia circular entre utils.py y server_handlers.py
async def send_notification(message, n_type='info'):
    """Envía una notificación a todos los clientes WebSocket conectados."""
    if not g_state['websockets']:
        return

    notification = {
        "type": "notification",
        "payload": {
            "message": message,
            "type": n_type
        }
    }
    
    # Usar asyncio.gather para enviar a todos los clientes en paralelo
    await asyncio.gather(
        *[ws.send_str(json.dumps(notification)) for ws in g_state['websockets']]
    )
