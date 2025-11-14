# src/server_handlers.py
import asyncio
import json
import logging
import os
from types import SimpleNamespace

from . import config as global_cfg
from .server_state import g_state

VENV_PYTHON_PATH = os.path.join(global_cfg.PROJECT_ROOT, "ath_venv", "bin", "python")

SAFE_CONFIG_KEYS = [
    'EXPERIMENT_NAME', 'LOAD_FROM_EXPERIMENT', 'CONTINUE_TRAINING',
    'MODEL_ARCHITECTURE', 'MODEL_PARAMS', 'GAMMA_DECAY', 'TOTAL_EPISODES', # ¡¡CAMBIO!!
    'STEPS_PER_EPISODE', 'LR_RATE_M', 'GRADIENT_CLIP', 'PESO_QUIETUD',
    'PESO_COMPLEJIDAD_LOCALIZADA', 'GRID_SIZE_TRAINING', 'GRID_SIZE_INFERENCE',
    'SAVE_EVERY_EPISODES', 'BATCH_SIZE_TRAINING', 'QCA_STEPS_TRAINING'
]

async def stream_process_output(stream, websocket, stream_name):
    # ... (código sin cambios)
    pass

async def create_experiment_handler(args):
    ws_id = args.pop('ws_id', None)
    websocket = g_state['websockets'].get(ws_id)
    if not websocket: return

    try:
        experiment_name = args.get("EXPERIMENT_NAME")
        if not experiment_name:
            await websocket.send_json({"type": "notification", "payload": {"status": "error", "message": "EXPERIMENT_NAME es requerido."}})
            return

        exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, experiment_name)
        config_path = os.path.join(exp_dir, "config.json")
        episodes_to_add = args.get('EPISODES_TO_ADD', global_cfg.TOTAL_EPISODES)

        if args.get('CONTINUE_TRAINING') and os.path.exists(config_path):
            logging.info(f"Continuando entrenamiento para '{experiment_name}'.")
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # --- ¡¡LÓGICA CORREGIDA!! Sumar al total de episodios ---
            current_total = config_data.get('TOTAL_EPISODES', 0)
            config_data['TOTAL_EPISODES'] = current_total + episodes_to_add
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            logging.info(f"Configuración actualizada. Nuevo objetivo: {config_data['TOTAL_EPISODES']} episodios.")

        else:
            os.makedirs(exp_dir, exist_ok=True)
            config_data = {key: getattr(global_cfg, key) for key in SAFE_CONFIG_KEYS if hasattr(global_cfg, key)}
            config_data.update(args)
            # --- ¡¡LÓGICA CORREGIDA!! Establecer el total inicial ---
            config_data['TOTAL_EPISODES'] = episodes_to_add
            
            config_to_save = {k: v for k, v in config_data.items() if k in SAFE_CONFIG_KEYS}
            if 'MODEL_PARAMS' in config_to_save and isinstance(config_to_save.get('MODEL_PARAMS'), SimpleNamespace):
                config_to_save['MODEL_PARAMS'] = vars(config_to_save['MODEL_PARAMS'])

            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            logging.info(f"Nueva config creada para '{experiment_name}' con objetivo de {episodes_to_add} episodios.")

        # ... (código para lanzar el subproceso sin cambios)
        script_path = os.path.join(global_cfg.PROJECT_ROOT, "scripts", "train.py")
        process = await asyncio.create_subprocess_exec(
            VENV_PYTHON_PATH, "-u", script_path, "--experiment_name", experiment_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        g_state['training_process'] = process
        asyncio.create_task(stream_process_output(process.stdout, websocket, 'stdout'))
        asyncio.create_task(stream_process_output(process.stderr, websocket, 'stderr'))
        await websocket.send_json({"type": "notification", "payload": {"status": "success", "message": f"Entrenamiento para '{experiment_name}' iniciado."}})

    except Exception as e:
        logging.error(f"Error al crear/continuar experimento: {e}", exc_info=True)
        if websocket and not websocket.closed:
            await websocket.send_json({"type": "notification", "payload": {"status": "error", "message": f"Error interno: {e}"}})