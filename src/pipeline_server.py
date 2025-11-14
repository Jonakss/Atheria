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
from .pipeline_viz import run_visualization_pipeline
from .model_loader import load_model
from .qca_engine import Aetheria_Motor

async def broadcast(data):
    """Envía datos a todos los websockets conectados."""
    for ws in list(g_state['websockets'].values()):
        if not ws.closed:
            await ws.send_json(data)

async def simulation_loop():
    """
    Bucle principal que ejecuta la simulación (inferencia) y envía los frames.
    """
    logging.info("Iniciando bucle de simulación (en pausa).")
    while True:
        try:
            if g_state.get('inference_running') and g_state.get('motor'):
                viz_type = g_state.get('viz_type', 'density_map')
                frame_data = run_visualization_pipeline(g_state['motor'], viz_type)
                if frame_data:
                    await broadcast({"type": "simulation_frame", "payload": frame_data})
            
            await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error en el bucle de simulación: {e}", exc_info=True)
            g_state['inference_running'] = False
            await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
            await asyncio.sleep(2)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    ws_id = str(uuid.uuid4())
    g_state['websockets'][ws_id] = ws
    logging.info(f"Cliente WebSocket conectado: {ws_id}. Total: {len(g_state['websockets'])}")

    try:
        initial_state = {
            "experiments": get_experiment_list(),
            "training_status": "running" if g_state.get('training_process') else "idle",
            "inference_status": "running" if g_state.get('inference_running') else "paused",
        }
        await ws.send_json({"type": "initial_state", "payload": initial_state})

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                scope, command, args = data.get("scope"), data.get("command"), data.get("args", {})
                args['ws_id'] = ws_id

                if scope == "experiment" and command == "create":
                    asyncio.create_task(create_experiment_handler(args))
                elif scope == "simulation" and command == "set_viz":
                    g_state['viz_type'] = args.get("viz_type", "density_map")
                elif scope == "inference" and command == "play":
                    if not g_state.get('motor'):
                        exp_list = get_experiment_list()
                        if exp_list:
                            config = load_experiment_config(exp_list[0]['name'])
                            model = load_model(config)
                            g_state['motor'] = Aetheria_Motor(model, global_cfg.GRID_SIZE_INFERENCE, config.MODEL_PARAMS.d_state, global_cfg.DEVICE)
                    g_state['inference_running'] = True
                    await broadcast({"type": "inference_status_update", "payload": {"status": "running"}})
                elif scope == "inference" and command == "pause":
                    g_state['inference_running'] = False
                    await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
                else:
                    await ws.send_json({"type": "notification", "payload": {"status": "error", "message": "Comando desconocido"}})
    finally:
        if ws_id in g_state['websockets']: del g_state['websockets'][ws_id]
        logging.info(f"Cliente WebSocket desconectado: {ws_id}. Total: {len(g_state['websockets'])}")
    return ws

async def http_handler(request):
    rel_path = request.match_info.get('tail', 'index.html')
    if not rel_path or rel_path == '/':
        rel_path = 'index.html'
    filepath = os.path.join(global_cfg.FRONTEND_DIST_PATH, rel_path)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        return web.FileResponse(filepath)
    index_path = os.path.join(global_cfg.FRONTEND_DIST_PATH, 'index.html')
    if os.path.exists(index_path):
        return web.FileResponse(index_path)
    return web.Response(status=404, text="Frontend no encontrado.")

def setup_routes(app):
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/{tail:.*}", http_handler)

async def on_startup(app):
    app['simulation_loop'] = asyncio.create_task(simulation_loop())

async def on_shutdown(app):
    app['simulation_loop'].cancel()
    try:
        await app['simulation_loop']
    except asyncio.CancelledError:
        pass

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
