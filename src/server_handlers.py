# src/server_handlers.py
import asyncio
import json
import logging
import os
from types import SimpleNamespace

from aiohttp import web

from . import config as global_cfg
from .server_state import g_state
from .utils import get_experiment_list, load_experiment_config

VENV_PYTHON_PATH = os.path.join(global_cfg.PROJECT_ROOT, "ath_venv", "bin", "python")

async def stream_process_output(stream, websocket, stream_name):
    """Lee la salida de un subproceso y la envía a un websocket específico."""
    while True:
        line = await stream.readline()
        if not line:
            break
        log_message = f"[{stream_name.upper()}] {line.decode().strip()}"
        # --- ¡¡CORRECCIÓN CLAVE!! Enviar solo al websocket correcto ---
        if not websocket.closed:
            await websocket.send_json({"type": "training_log", "payload": log_message})
        else:
            break # Detener si el websocket se ha cerrado

async def create_experiment_handler(args):
    """
    Maneja la creación de un nuevo experimento.
    Ahora los logs se envían solo al cliente que lo solicitó.
    """
    ws_id = args.pop('ws_id', None)
    if not ws_id or ws_id not in g_state['websockets']:
        logging.error(f"Intento de crear experimento con ws_id inválido: {ws_id}")
        return

    websocket = g_state['websockets'][ws_id]

    try:
        # Construcción dinámica de la configuración
        config_data = {key: getattr(global_cfg, key) for key in dir(global_cfg) if key.isupper()}
        config_data.update(args)
        
        if 'MODEL_PARAMS' in config_data and isinstance(config_data['MODEL_PARAMS'], dict):
            config_data['MODEL_PARAMS'] = SimpleNamespace(**config_data['MODEL_PARAMS'])

        experiment_name = config_data.get("EXPERIMENT_NAME")
        if not experiment_name:
            await websocket.send_json({"type": "notification", "payload": {"status": "error", "message": "EXPERIMENT_NAME es requerido."}})
            return

        exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)

        config_path = os.path.join(exp_dir, "config.json")
        config_to_save = config_data.copy()
        if 'MODEL_PARAMS' in config_to_save and isinstance(config_to_save['MODEL_PARAMS'], SimpleNamespace):
            config_to_save['MODEL_PARAMS'] = vars(config_to_save['MODEL_PARAMS'])
        
        # --- ¡¡CORRECCIÓN CLAVE!! Eliminar objetos no serializables ---
        config_to_save.pop('DEVICE', None)

        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)

        script_path = os.path.join(global_cfg.PROJECT_ROOT, "scripts", "train.py")
        process = await asyncio.create_subprocess_exec(
            VENV_PYTHON_PATH, script_path, experiment_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        g_state['training_process'] = process

        # Iniciar tareas para streamear stdout y stderr al websocket correcto
        asyncio.create_task(stream_process_output(process.stdout, websocket, 'stdout'))
        asyncio.create_task(stream_process_output(process.stderr, websocket, 'stderr'))

        await websocket.send_json({"type": "notification", "payload": {"status": "success", "message": f"Experimento '{experiment_name}' creado y entrenamiento iniciado."}})

    except Exception as e:
        logging.error(f"Error al crear el experimento para {ws_id}: {e}", exc_info=True)
        if not websocket.closed:
            await websocket.send_json({"type": "notification", "payload": {"status": "error", "message": f"Error interno: {e}"}})