# src/pipeline_server.py
import asyncio
import json
import logging
import os
import uuid
from aiohttp import web

from . import config as global_cfg
from .server_state import g_state
from .utils import get_experiment_list, load_experiment_config
from .server_handlers import create_experiment_handler
from .pipeline_viz import get_visualization_data
from .model_loader import load_model
from .qca_engine import Aetheria_Motor, QuantumState

async def broadcast(data):
    for ws in list(g_state['websockets'].values()):
        if not ws.closed: await ws.send_json(data)

async def simulation_loop():
    logging.info("Iniciando bucle de simulación (en pausa).")
    while True:
        try:
            if not g_state.get('is_paused', True) and g_state.get('motor'):
                g_state['motor'].evolve_internal_state() # ¡¡CORRECCIÓN!!
                viz_data = get_visualization_data(g_state['motor'].state.psi, g_state['viz_type'])
                
                frame_payload = {
                    "step": g_state['simulation_step'],
                    "map_data": viz_data["map_data"],
                    "hist_data": viz_data["hist_data"]
                }
                
                tasks = [ws.send_json({"type": "simulation_frame", "payload": frame_payload}) for ws in g_state['websockets'].values()]
                await asyncio.gather(*tasks)
                
                g_state['simulation_step'] += 1
            await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error en el bucle de simulación: {e}", exc_info=True)
            g_state['is_paused'] = True # ¡¡CORRECCIÓN!!
            await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
            await asyncio.sleep(2)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_id = str(uuid.uuid4())
    g_state['websockets'][ws_id] = ws
    logging.info(f"Cliente WebSocket conectado: {ws_id}. Total: {len(g_state['websockets'])}")
    try:
        initial_state = {"experiments": get_experiment_list(), "training_status": "idle", "inference_status": "paused"}
        await ws.send_json({"type": "initial_state", "payload": initial_state})
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                scope, cmd, args = data.get("scope"), data.get("command"), data.get("args", {})
                args['ws_id'] = ws_id
                if scope == "experiment" and cmd == "create": 
                    args['CONTINUE_TRAINING'] = False
                    asyncio.create_task(create_experiment_handler(args))
                elif scope == "experiment" and cmd == "continue":
                    args['CONTINUE_TRAINING'] = True
                    asyncio.create_task(create_experiment_handler(args))
                elif scope == "experiment" and cmd == "stop":
                    logging.info(f"Recibida orden de detener el entrenamiento para el cliente {ws_id}.")
                    if g_state.get('training_process'):
                        try:
                            g_state['training_process'].kill()
                            await g_state['training_process'].wait()
                            g_state['training_process'] = None
                            await ws.send_json({"type": "training_status_update", "payload": {"status": "idle"}})
                            await ws.send_json({"type": "notification", "payload": {"status": "info", "message": "Entrenamiento detenido por el usuario."}})
                        except ProcessLookupError:
                            g_state['training_process'] = None
                            await ws.send_json({"type": "training_status_update", "payload": {"status": "idle"}})
                elif scope == "simulation" and cmd == "set_viz": g_state['viz_type'] = args.get("viz_type", "density_map")
                elif scope == "inference" and cmd == "play":
                    g_state['is_paused'] = False
                    await broadcast({"type": "inference_status_update", "payload": {"status": "running"}})
                elif scope == "inference" and cmd == "pause":
                    g_state['is_paused'] = True
                    await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
                # --- ¡¡NUEVO!! Handlers de Herramientas de Inferencia ---
                elif scope == "inference" and cmd == "load_experiment":
                    exp_name = args.get("experiment_name")
                    if exp_name:
                        config = load_experiment_config(exp_name)
                        if config:
                            model = load_model(config)
                            d_state = config.MODEL_PARAMS.d_state
                            g_state['motor'] = Aetheria_Motor(model, global_cfg.GRID_SIZE_INFERENCE, d_state, global_cfg.DEVICE)
                            logging.info(f"Motor de inferencia cargado con el modelo de: {exp_name}")
                            await ws.send_json({"type": "notification", "payload": {"status": "success", "message": f"Modelo '{exp_name}' cargado."}})
                elif scope == "inference" and cmd == "reset":
                    if g_state.get('motor'):
                        g_state['motor'].state = QuantumState(g_state['motor'].grid_size, g_state['motor'].d_state, g_state['motor'].device)
                        logging.info("Estado de la simulación reiniciado.")
    finally:
        if ws_id in g_state['websockets']: del g_state['websockets'][ws_id]
        logging.info(f"Cliente WebSocket desconectado: {ws_id}. Total: {len(g_state['websockets'])}")
    return ws

async def http_handler(request):
    rel_path = request.match_info.get('tail', 'index.html')
    filepath = os.path.join(global_cfg.FRONTEND_DIST_PATH, rel_path or 'index.html')
    if os.path.exists(filepath) and os.path.isfile(filepath): return web.FileResponse(filepath)
    index_path = os.path.join(global_cfg.FRONTEND_DIST_PATH, 'index.html')
    if os.path.exists(index_path): return web.FileResponse(index_path)
    return web.Response(status=404, text="Frontend no encontrado.")

def setup_routes(app):
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/{tail:.*}", http_handler)

async def on_startup(app): app['simulation_loop'] = asyncio.create_task(simulation_loop())
async def on_shutdown(app):
    app['simulation_loop'].cancel()
    try: await app['simulation_loop']
    except asyncio.CancelledError: pass

async def main():
    app = web.Application()
    setup_routes(app)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, global_cfg.LAB_SERVER_HOST, global_cfg.LAB_SERVER_PORT)
    logging.info(f"Servidor iniciado en http://{global_cfg.LAB_SERVER_HOST}:{global_cfg.LAB_SERVER_PORT}")
    await site.start()
    await asyncio.Event().wait()