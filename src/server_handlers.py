# src/server_handlers.py
import asyncio
import json
import logging
import os
import glob
import base64
from io import BytesIO
import numpy as np
import torch
from aiohttp import web
from PIL import Image
from scipy.stats import entropy

from . import config as cfg
from .qca_engine import Aetheria_Motor
from .model_loader import load_model
from .server_state import g_state
from .visualization_tools import (
    get_density_map, get_change_map, get_phase_map, 
    get_channels_map, fig_to_base64
)

# --- Lógica de Métricas ---
def calculate_metrics(psi_tensor):
    if psi_tensor is None: return {}
    density_grid = get_density_map(psi_tensor)
    density_flat = density_grid.flatten()
    
    density_sum = np.sum(density_flat)
    if density_sum == 0: return {"entropy": 0, "stability_l2": 0, "complexity_lz": 0}
    
    pk = density_flat / density_sum
    
    # L2 Stability (lower is more stable)
    stability_l2 = np.sqrt(np.mean(np.square(density_flat)))

    return {
        "entropy": entropy(pk).item(),
        "stability_l2": stability_l2.item(),
        "complexity_lz": 0 # Placeholder for future LZ complexity calculation
    }

# --- Lógica de Visualización ---
def get_visualization_as_base64(psi, prev_psi, viz_type):
    if psi is None:
        img = Image.new('RGB', (256, 256), color='black')
    else:
        if viz_type == 'density':
            grid = get_density_map(psi)
        elif viz_type == 'change':
            grid = get_change_map(psi, prev_psi) if prev_psi is not None else np.zeros_like(get_density_map(psi))
        elif viz_type == 'phase':
            grid = get_phase_map(psi)
        elif viz_type == 'channels':
            grid = get_channels_map(psi)
        else:
            grid = get_density_map(psi) # Default
        
        # Convertir a imagen
        grid_normalized = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8) * 255
        img = Image.fromarray(grid_normalized.astype(np.uint8))

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# --- Bucles Asíncronos ---
async def simulation_loop():
    logging.info("Iniciando bucle de simulación...")
    while g_state.sim_motor is not None:
        try:
            if not g_state.sim_is_paused:
                g_state.sim_previous_psi = g_state.sim_latest_psi.clone()
                g_state.sim_motor.evolve_step()
                g_state.sim_latest_psi = g_state.sim_motor.state.psi.clone()
                g_state.sim_step_count += 1
            await asyncio.sleep(1 / 120) # Mayor frecuencia de simulación
        except asyncio.CancelledError:
            break
    logging.info("Bucle de simulación detenido.")

async def broadcast_loop():
    logging.info("Iniciando bucle de broadcast...")
    while True:
        try:
            # Enviar estado y métricas a una frecuencia más baja
            if g_state.clients:
                g_state.sim_latest_metrics = calculate_metrics(g_state.sim_latest_psi)
                status_payload = {
                    "type": "status_update",
                    "payload": {
                        "status": g_state.get_sim_status(),
                        "step_count": g_state.sim_step_count,
                        "config": g_state.sim_config,
                    }
                }
                metrics_payload = {"type": "metrics_update", "payload": g_state.sim_latest_metrics}
                await asyncio.gather(
                    broadcast_message(status_payload),
                    broadcast_message(metrics_payload)
                )

            # Enviar imagen a una frecuencia más alta
            if g_state.clients and g_state.get_sim_status() == 'running':
                img_payload = {
                    "type": "image_update",
                    "payload": get_visualization_as_base64(
                        g_state.sim_latest_psi, 
                        g_state.sim_previous_psi, 
                        g_state.sim_viz_type
                    )
                }
                await broadcast_message(img_payload)

            await asyncio.sleep(1 / 30) # Frecuencia de broadcast
        except asyncio.CancelledError:
            break
    logging.info("Bucle de broadcast detenido.")

# --- Funciones de Utilidad y Control ---
async def broadcast_message(message: dict):
    if g_state.clients:
        await asyncio.gather(*[ws.send_json(message) for ws in list(g_state.clients) if not ws.closed])

def get_checkpoints():
    # ... (código sin cambios)
    pass # Placeholder

async def start_simulation(args):
    # ... (código sin cambios)
    pass # Placeholder

def stop_simulation():
    # ... (código sin cambios)
    pass # Placeholder

# --- Manejador de WebSocket Principal ---
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    g_state.clients.add(ws)
    logging.info(f"Cliente conectado. Clientes totales: {len(g_state.clients)}")

    await ws.send_json({
        "type": "checkpoints_update",
        "payload": {"checkpoints": get_checkpoints()}
    })

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    scope = data.get("scope")
                    command = data.get("command")
                    args = data.get("args", {})

                    if scope == "sim":
                        if command == "start": await start_simulation(args)
                        elif command == "stop": stop_simulation()
                        elif command == "pause": g_state.sim_is_paused = True
                        elif command == "resume": g_state.sim_is_paused = False
                        elif command == "set_viz": g_state.sim_viz_type = args.get('type', 'density')
                    
                    elif scope == "lab":
                        if command == "refresh_checkpoints":
                            await ws.send_json({
                                "type": "checkpoints_update",
                                "payload": {"checkpoints": get_checkpoints()}
                            })
                        # elif command == "start_training": await start_training(args)
                        # elif command == "stop_training": stop_training()

                except json.JSONDecodeError:
                    logging.warning("Mensaje WebSocket no es un JSON válido.")
                except Exception as e:
                    logging.error(f"Error procesando comando: {e}", exc_info=True)
    finally:
        g_state.clients.remove(ws)
        logging.info(f"Cliente desconectado. Clientes restantes: {len(g_state.clients)}")
    return ws
