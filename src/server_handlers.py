# src/server_handlers.py
import asyncio
import logging
import json
import os
from aiohttp import web
from . import config as cfg
from .server_state import g_state, send_notification
from .utils import get_experiment_list, get_latest_checkpoint, save_experiment_config, load_experiment_config
from .model_loader import load_model
from .pipeline_viz import VisualizationPipeline

g_command_handlers = {}

def register_handler(scope, command):
    def decorator(func):
        g_command_handlers[(scope, command)] = func
        return func
    return decorator

async def stream_process_output(process):
    async def read_stream(stream, stream_name):
        while not stream.at_eof():
            line = await stream.readline()
            if line:
                line_str = line.decode('utf-8').strip()
                log_message = {"type": "training_log", "payload": line_str}
                await asyncio.gather(*[ws.send_str(json.dumps(log_message)) for ws in g_state['websockets']])

    await asyncio.gather(
        read_stream(process.stdout, 'stdout'),
        read_stream(process.stderr, 'stderr')
    )
    
    await process.wait()
    g_state['training']['status'] = 'finished' if process.returncode == 0 else 'error'
    await send_notification(f"Entrenamiento finalizado con código {process.returncode}", g_state['training']['status'])

@register_handler('sim', 'start')
async def start_simulation_handler(args):
    exp_name = args.get('experiment_name')
    if not exp_name: raise ValueError("Se requiere 'experiment_name' para iniciar la simulación.")
    g_state['simulation']['is_running'] = True
    g_state['simulation']['current_experiment'] = exp_name
    g_state['simulation']['current_step'] = 0
    exp_config = load_experiment_config(exp_name)
    latest_model_chk = get_latest_checkpoint(exp_name, "qca")
    if not latest_model_chk: raise FileNotFoundError(f"No se encontraron checkpoints para '{exp_name}'.")
    model = load_model(exp_config, latest_model_chk)
    g_state['simulation']['pipeline'] = VisualizationPipeline(model, exp_config)
    await send_notification(f"Simulación de '{exp_name}' iniciada.", "success")

@register_handler('sim', 'stop')
async def stop_simulation_handler(args):
    g_state['simulation']['is_running'] = False
    g_state['simulation']['pipeline'] = None
    exp_name = g_state['simulation']['current_experiment']
    await send_notification(f"Simulación de '{exp_name}' detenida.", "info")

@register_handler('lab', 'create_experiment')
async def create_experiment_handler(args):
    params = args.get('params', {})
    exp_name = params.get('experiment_name')
    if not exp_name: raise ValueError("El nombre del experimento no puede estar vacío.")

    config_data = {
        'CONTINUE_TRAINING': params.get('continue_training', False),
        'EPISODES_TO_ADD': cfg.EPISODES_TO_ADD, 'STEPS_PER_EPISODE': cfg.STEPS_PER_EPISODE,
        'SAVE_EVERY_EPISODES': cfg.SAVE_EVERY_EPISODES, 'GRADIENT_CLIP': cfg.GRADIENT_CLIP,
        'PESO_QUIETUD': cfg.PESO_QUIETUD, 'PESO_COMPLEJIDAD_LOCALIZADA': cfg.PESO_COMPLEJIDAD_LOCALIZADA,
        'GRID_SIZE_INFERENCE': cfg.GRID_SIZE_INFERENCE, 'GAMMA_DECAY': cfg.GAMMA_DECAY,
        'experiment_name': exp_name,
        'MODEL_ARCHITECTURE': params.get('model_architecture') or cfg.MODEL_ARCHITECTURE,
        'LR_RATE_M': params.get('lr_rate_m') or cfg.LR_RATE_M,
        'GRID_SIZE_TRAINING': params.get('grid_size_training') or cfg.GRID_SIZE_TRAINING,
        'MODEL_PARAMS': {
            'd_state': params.get('d_state') or cfg.MODEL_PARAMS['d_state'],
            'hidden_channels': params.get('hidden_channels') or cfg.MODEL_PARAMS['hidden_channels'],
            'alpha': params.get('alpha') or cfg.MODEL_PARAMS.get('alpha', 0.9),
            'beta': params.get('beta') or cfg.MODEL_PARAMS.get('beta', 0.85),
        }
    }
    save_experiment_config(exp_name, config_data)
    logging.info(f"Configuración para '{exp_name}' guardada.")
    
    # --- ¡¡SOLUCIÓN DEFINITIVA!! Usar el Python del venv ---
    script_path = os.path.join(cfg.PROJECT_ROOT, 'scripts', 'train.py')
    python_executable = os.path.join(cfg.PROJECT_ROOT, 'ath_venv', 'bin', 'python')

    process = await asyncio.create_subprocess_exec(
        python_executable, script_path, '--experiment_name', exp_name,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    
    g_state['training']['process'] = process
    g_state['training']['current_experiment'] = exp_name
    g_state['training']['status'] = 'running'
    await send_notification(f"Entrenamiento de '{exp_name}' iniciado...", "info")
    asyncio.create_task(stream_process_output(process))

async def main_command_handler(scope, command, args):
    handler = g_command_handlers.get((scope, command))
    if handler:
        try:
            await handler(args)
        except Exception as e:
            error_message = f"Error al procesar '{command}': {e}"
            logging.error(error_message, exc_info=True)
            await send_notification(f"Error en Servidor - {error_message}", "error")
    else:
        error_message = f"Comando desconocido: {scope}.{command}"
        logging.warning(error_message)
        await send_notification(error_message, "warning")

async def simulation_loop():
    while True:
        if g_state['simulation']['is_running'] and g_state['simulation']['pipeline']:
            try:
                viz_pipeline = g_state['simulation']['pipeline']
                frame_data = viz_pipeline.run_step()
                message = {
                    "type": "simulation_frame",
                    "payload": {
                        "experiment_name": g_state['simulation']['current_experiment'],
                        "step": viz_pipeline.step_count,
                        "frame_data": frame_data
                    }
                }
                if g_state['websockets']:
                    await asyncio.gather(*[ws.send_str(json.dumps(message)) for ws in g_state['websockets']])
            except Exception as e:
                logging.error(f"Error en el bucle de simulación: {e}", exc_info=True)
                await send_notification(f"Error en Simulación: {e}", "error")
                g_state['simulation']['is_running'] = False
        await asyncio.sleep(1.0 / g_state['simulation']['fps'])

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    g_state['websockets'].append(ws)
    logging.info(f"Cliente WebSocket conectado. Total: {len(g_state['websockets'])}")
    try:
        initial_state = {
            "type": "initial_state",
            "payload": {
                "experiments": get_experiment_list(),
                "training_status": g_state['training']['status'],
                "simulation_status": g_state['simulation']['is_running'],
            }
        }
        await ws.send_str(json.dumps(initial_state))
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    scope, command, args = data.get('scope'), data.get('command'), data.get('args', {})
                    await main_command_handler(scope, command, args)
                except Exception as e:
                    logging.error(f"Error procesando mensaje: {e}", exc_info=True)
            elif msg.type == web.WSMsgType.ERROR:
                logging.error(f'Conexión cerrada con excepción {ws.exception()}')
    finally:
        g_state['websockets'].remove(ws)
        logging.info(f"Cliente WebSocket desconectado. Total: {len(g_state['websockets'])}")
    return ws