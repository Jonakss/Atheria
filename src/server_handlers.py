# src/server_handlers.py
import asyncio
import json
import logging
import os
from types import SimpleNamespace

from . import config as global_cfg
from .server_state import g_state

VENV_PYTHON_PATH = os.path.join(global_cfg.PROJECT_ROOT, "ath_venv", "bin", "python")

# --- ¡¡NUEVO!! Lista explícita de claves de configuración seguras para guardar ---
SAFE_CONFIG_KEYS = [
    'EXPERIMENT_NAME', 'LOAD_FROM_EXPERIMENT', 'CONTINUE_TRAINING',
    'MODEL_ARCHITECTURE', 'MODEL_PARAMS', 'GAMMA_DECAY', 'EPISODES_TO_ADD',
    'STEPS_PER_EPISODE', 'LR_RATE_M', 'GRADIENT_CLIP', 'PESO_QUIETUD',
    'PESO_COMPLEJIDAD_LOCALIZADA', 'GRID_SIZE_TRAINING', 'GRID_SIZE_INFERENCE',
    'SAVE_EVERY_EPISODES'
]

async def stream_process_output(stream, websocket, stream_name):
    while True:
        line = await stream.readline()
        if not line: break
        log_message = f"[{stream_name.upper()}] {line.decode().strip()}"
        if not websocket.closed:
            await websocket.send_json({"type": "training_log", "payload": log_message})
        else:
            break

async def create_experiment_handler(args):
    ws_id = args.pop('ws_id', None)
    if not ws_id or ws_id not in g_state['websockets']:
        logging.error(f"Intento de crear experimento con ws_id inválido: {ws_id}")
        return
    websocket = g_state['websockets'][ws_id]

    try:
        # --- ¡¡REFACTORIZACIÓN CLAVE!! Construcción dinámica y SEGURA de la configuración ---
        # 1. Cargar solo los defaults seguros de src/config.py
        config_data = {key: getattr(global_cfg, key) for key in SAFE_CONFIG_KEYS if hasattr(global_cfg, key)}

        # 2. Sobrescribir con los valores recibidos del frontend
        config_data.update(args)
        
        experiment_name = config_data.get("EXPERIMENT_NAME")
        if not experiment_name:
            await websocket.send_json({"type": "notification", "payload": {"status": "error", "message": "EXPERIMENT_NAME es requerido."}})
            return

        exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        config_path = os.path.join(exp_dir, "config.json")
        
        # config_data ya es seguro para guardar, solo hay que manejar SimpleNamespace
        config_to_save = config_data.copy()
        if 'MODEL_PARAMS' in config_to_save and isinstance(config_to_save.get('MODEL_PARAMS'), SimpleNamespace):
            config_to_save['MODEL_PARAMS'] = vars(config_to_save['MODEL_PARAMS'])

        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)

        script_path = os.path.join(global_cfg.PROJECT_ROOT, "scripts", "train.py")
        process = await asyncio.create_subprocess_exec(
            VENV_PYTHON_PATH, script_path, experiment_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        g_state['training_process'] = process

        asyncio.create_task(stream_process_output(process.stdout, websocket, 'stdout'))
        asyncio.create_task(stream_process_output(process.stderr, websocket, 'stderr'))

        await websocket.send_json({"type": "notification", "payload": {"status": "success", "message": f"Experimento '{experiment_name}' creado y entrenamiento iniciado."}})

    except Exception as e:
        logging.error(f"Error al crear el experimento para {ws_id}: {e}", exc_info=True)
        if not websocket.closed:
            await websocket.send_json({"type": "notification", "payload": {"status": "error", "message": f"Error interno: {e}"}})
