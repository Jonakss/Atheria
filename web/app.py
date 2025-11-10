# app.py - Servidor Unificado de Aetheria
import asyncio
import os
import sys
import glob
import json
import logging
from aiohttp import web
import websockets # Importar websockets para ConnectionClosed

# --- Configuración del Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Importaciones del Proyecto ---
try:
    from src import config as cfg
    from src.qca_engine import Aetheria_Motor
    from src.model_loader import load_model
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import io
    import base64
    from collections import deque
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import entropy
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('dark_background')

# --- Estado Global de la Aplicación ---
g_state = {
    "training_process": None,
    "simulation_tasks": [], # Lista de asyncio.Task para sim_loop y broadcast_loop
    "clients": set(), # Clientes conectados al WebSocket principal
    
    # Estado de la Simulación
    "sim_is_paused": False,
    "sim_viz_type": 'density',
    "sim_latest_psi": None,
    "sim_previous_psi": None,
    "sim_step_count": 0,
    "sim_density_history": deque(maxlen=2000),
    "sim_spacetime_buffer": deque(maxlen=cfg.GRID_SIZE),
    "sim_cube_buffer": deque(maxlen=cfg.STEPS_PER_EPISODE), # <-- ¡NUEVO! Para el cubo 3D
    "sim_latest_metrics": {},
    "sim_connected_clients": {}, # {ws_aiohttp: {"viewport": ...}}
    "sim_viewport_cache": {},
    "sim_motor": None,
    "sim_model": None,
}
DEFAULT_VIEWPORT = {"x": 0, "y": 0, "width": 1, "height": 1}

# --- Lógica de Visualización y Métricas ---
def get_complex_parts(psi_tensor):
    """Helper para dividir un tensor real en partes real e imaginaria."""
    # psi_tensor shape: [B, C, H, W]
    num_dims = psi_tensor.shape[1]
    d_state = num_dims // 2
    real_parts = psi_tensor[:, :d_state, :, :]
    imag_parts = psi_tensor[:, d_state:, :, :]
    return real_parts, imag_parts

def calculate_metrics(psi_tensor):
    if psi_tensor is None: return {}
    real_parts, imag_parts = get_complex_parts(psi_tensor)
    density_grid = real_parts.pow(2) + imag_parts.pow(2) # Shape [B, D_STATE, H, W]
    density_flat = torch.sum(density_grid, dim=1).squeeze(0).cpu().numpy().flatten() # Sum over D_STATE, then flatten
    
    density_sum = np.sum(density_flat)
    if density_sum == 0: return {"mean_density": 0, "variance": 0, "entropy": 0}
    pk = density_flat / density_sum
    return {
        "mean_density": np.mean(density_flat).item(),
        "variance": np.var(density_flat).item(),
        "entropy": entropy(pk).item()
    }

def generate_view_tensor(psi_tensor, viewport):
    """Generates a view of the tensor based on the viewport using grid_sample."""
    if psi_tensor is None: return None
    
    # psi_tensor is [B, C, H, W]
    _, _, h, w = psi_tensor.shape
    
    v_x, v_y, v_w, v_h = viewport['x'], viewport['y'], viewport['width'], viewport['height']

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(v_y, v_y + v_h, steps=h, device=cfg.DEVICE),
        torch.linspace(v_x, v_x + v_w, steps=w, device=cfg.DEVICE),
        indexing='ij'
    )

    sample_x = grid_x * 2 - 1
    sample_y = grid_y * 2 - 1

    sample_grid = torch.stack((sample_x, sample_y), dim=-1).unsqueeze(0)

    view_tensor = F.grid_sample(
        psi_tensor,
        sample_grid,
        mode='bicubic',
        padding_mode='zeros',
        align_corners=False
    )
    return view_tensor

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def generate_poincare_plot():
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(g_state["sim_density_history"]) > 1: ax.scatter(list(g_state["sim_density_history"])[:-1], list(g_state["sim_density_history"])[1:], alpha=0.5, s=5, color='cyan')
    ax.set_title('Gráfico de Poincaré'); ax.set_xlabel('Densidad en t'); ax.set_ylabel('Densidad en t+1'); ax.grid(True, alpha=0.2)
    return fig

def generate_density_histogram(psi_tensor):
    fig, ax = plt.subplots(figsize=(6, 4))
    if psi_tensor is not None:
        real_parts, imag_parts = get_complex_parts(psi_tensor)
        densities = torch.sum(real_parts.pow(2) + imag_parts.pow(2), dim=1).squeeze(0).cpu().numpy().flatten()
        ax.hist(densities, bins=50, color='deepskyblue', alpha=0.8)
    ax.set_title('Histograma de Densidad'); ax.set_xlabel('Densidad'); ax.set_ylabel('Frecuencia'); ax.grid(True, alpha=0.2)
    return fig

def generate_spacetime_plot():
    fig, ax = plt.subplots(figsize=(6, 6))
    if g_state["sim_spacetime_buffer"]:
        # Normalizar el buffer para visualización
        buffer_np = np.array(g_state["sim_spacetime_buffer"])
        max_val = np.max(buffer_np)
        if max_val > 0: buffer_np = buffer_np / max_val
        ax.imshow(buffer_np, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title('Diagrama Espacio-Tiempo (Fila Central)'); ax.set_xlabel('Posición en la Fila'); ax.set_ylabel('Tiempo (Pasos)')
    return fig

def generate_spacetime_cube_plot():
    """Genera una visualización de cubo 3D de los últimos 50 pasos."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if not g_state["sim_cube_buffer"]:
        ax.set_title("Buffer de Cubo Espacio-Tiempo Vacío")
        return fig_to_base64(fig)

    all_x, all_y, all_t, all_colors = [], [], [], []
    
    # Umbral para visualizar solo las celdas "activas"
    density_threshold = 0.1

    for t, psi_t in enumerate(g_state["sim_cube_buffer"]):
        real_parts, imag_parts = get_complex_parts(psi_t)
        density_grid = torch.sum(real_parts.pow(2) + imag_parts.pow(2), dim=1).squeeze(0).cpu().numpy()
        
        # Encontrar coordenadas por encima del umbral
        y_coords, x_coords = np.where(density_grid > density_threshold)
        
        if len(x_coords) > 0:
            densities = density_grid[y_coords, x_coords]
            all_x.extend(x_coords)
            all_y.extend(y_coords)
            all_t.extend([t] * len(x_coords))
            all_colors.extend(densities)

    if all_x:
        scatter = ax.scatter(all_x, all_y, all_t, c=all_colors, cmap='viridis', s=2, alpha=0.7)
        fig.colorbar(scatter, ax=ax, label='Densidad de Estado')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(f'Tiempo (Últimos {cfg.STEPS_PER_EPISODE} pasos)')
    ax.set_title('Cubo Espacio-Tiempo')
    ax.view_init(elev=20, azim=-60)
    
    return fig_to_base64(fig)

def get_visualization_as_base64(psi, prev_psi, viz_type, viewport):
    # Visualizaciones que no son de grid y no se recortan
    if viz_type == 'poincare': return fig_to_base64(generate_poincare_plot())
    if viz_type == 'density_histogram': return fig_to_base64(generate_density_histogram(psi))
    if viz_type == 'spacetime_slice': return fig_to_base64(generate_spacetime_plot())
    if viz_type == 'spacetime_cube': return generate_spacetime_cube_plot()
    
    # Aplicar viewport para visualizaciones de grid
    psi_to_render = generate_view_tensor(psi, viewport)
    if psi_to_render is None: return get_black_screen_base64()

    real_parts_render, imag_parts_render = get_complex_parts(psi_to_render)
    
    # --- Cálculos de Grid ---
    if viz_type == 'state_change':
        prev_psi_to_render = generate_view_tensor(prev_psi, viewport)
        if prev_psi_to_render is None: return get_black_screen_base64()
        
        # Calcular el cambio en la magnitud de los vectores complejos
        change_sq = (psi_to_render - prev_psi_to_render).abs().pow(2)
        image_data = torch.sum(change_sq, dim=1).squeeze(0).cpu().numpy()
    elif viz_type == 'density':
        image_data = torch.sum(real_parts_render.pow(2) + imag_parts_render.pow(2), dim=1).squeeze(0).cpu().numpy()
    elif viz_type == 'channels':
        # Visualizar los primeros 3 canales complejos como RGB
        num_complex_channels = real_parts_render.shape[1]
        if num_complex_channels >= 3:
            r = (real_parts_render[:, 0, :, :].pow(2) + imag_parts_render[:, 0, :, :].pow(2)).squeeze(0).cpu().numpy()
            g = (real_parts_render[:, 1, :, :].pow(2) + imag_parts_render[:, 1, :, :].pow(2)).squeeze(0).cpu().numpy()
            b = (real_parts_render[:, 2, :, :].pow(2) + imag_parts_render[:, 2, :, :].pow(2)).squeeze(0).cpu().numpy()
            image_data = np.stack([r, g, b], axis=-1)
        else: # Si hay menos de 3 canales, usa lo que hay y rellena con ceros
            channels_data = []
            for i in range(num_complex_channels):
                channels_data.append((real_parts_render[:, i, :, :].pow(2) + imag_parts_render[:, i, :, :].pow(2)).squeeze(0).cpu().numpy())
            while len(channels_data) < 3:
                channels_data.append(np.zeros_like(channels_data[0]))
            image_data = np.stack(channels_data, axis=-1)
    elif viz_type == 'aggregate_phase':
        sum_real = torch.sum(real_parts_render, dim=1).squeeze(0).cpu().numpy()
        sum_imag = torch.sum(imag_parts_render, dim=1).squeeze(0).cpu().numpy()
        image_data = np.angle(sum_real + 1j * sum_imag)
    elif viz_type == 'fft':
        density_grid = torch.sum(real_parts_render.pow(2) + imag_parts_render.pow(2), dim=1).squeeze(0).cpu().numpy()
        fft_result = np.fft.fft2(density_grid)
        fft_shifted = np.fft.fftshift(fft_result)
        image_data = np.log(np.abs(fft_shifted) + 1)
    else: # Fallback a densidad
        image_data = torch.sum(real_parts_render.pow(2) + imag_parts_render.pow(2), dim=1).squeeze(0).cpu().numpy()

    # --- Normalización y renderizado de imagen ---
    max_val = np.max(image_data)
    normalized_data = image_data / max_val if max_val > 0 else image_data
    scaled_data = (normalized_data * 255).astype(np.uint8)
    img = Image.fromarray(scaled_data, 'L' if scaled_data.ndim == 2 else 'RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def get_black_screen_base64():
    img = Image.new('L', (cfg.GRID_SIZE, cfg.GRID_SIZE), color=0)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

# --- Bucles de Simulación ---
async def simulation_loop():
    logging.info("Iniciando bucle de simulación...")
    while True:
        try:
            if not g_state["sim_is_paused"]:
                g_state["sim_previous_psi"] = g_state["sim_latest_psi"].clone()
                g_state["sim_motor"].evolve_step()
                g_state["sim_motor"].state.normalize_() # Normalizar explícitamente
                g_state["sim_latest_psi"] = g_state["sim_motor"].state.psi.clone()
                
                g_state["sim_step_count"] += 1
                g_state["sim_latest_metrics"] = calculate_metrics(g_state["sim_latest_psi"])
                g_state["sim_density_history"].append(g_state["sim_latest_metrics"].get("mean_density", 0))
                
                # Actualizar buffer espacio-tiempo
                center_row_density = torch.sum(real_parts.pow(2) + imag_parts.pow(2), dim=1).squeeze(0)[cfg.GRID_SIZE // 2, :].cpu().numpy()
                g_state["sim_spacetime_buffer"].append(center_row_density)
                g_state["sim_cube_buffer"].append(g_state["sim_latest_psi"].clone()) # <-- ¡NUEVO!
            await asyncio.sleep(1 / 60)
        except asyncio.CancelledError:
            logging.info("Bucle de simulación detenido.")
            break
        except Exception as e:
            logging.error(f"Error en simulation_loop: {e}", exc_info=True)
            await asyncio.sleep(1)

async def broadcast_loop():
    logging.info("Iniciando bucle de broadcast...")
    while True:
        try:
            await asyncio.sleep(1 / 30)
            if not g_state["sim_connected_clients"]: continue
            
            g_state["sim_viewport_cache"].clear()
            clients_to_update = list(g_state["sim_connected_clients"].items())

            for ws, state in clients_to_update:
                try:
                    viewport_tuple = tuple(state['viewport'].items())
                    if viewport_tuple in g_state["sim_viewport_cache"]:
                        image_data = g_state["sim_viewport_cache"][viewport_tuple]
                    else:
                        image_data = get_visualization_as_base64(
                            g_state["sim_latest_psi"], 
                            g_state["sim_previous_psi"], 
                            g_state["sim_viz_type"], 
                            state['viewport']
                        )
                        g_state["sim_viewport_cache"][viewport_tuple] = image_data
                    
                    payload = {
                        'type': 'sim_update',
                        'step': g_state["sim_step_count"],
                        'frame_type': g_state["sim_viz_type"],
                        'image_data': image_data,
                        'metrics': g_state["sim_latest_metrics"]
                    }
                    await ws.send_json(payload)
                except ConnectionResetError:
                    logging.warning(f"Error de conexión al enviar a {ws.remote_address}. El cliente será eliminado.")
                except Exception as e:
                    logging.error(f"Error en sub-bucle de broadcast: {e}", exc_info=True)
        except asyncio.CancelledError:
            logging.info("Bucle de broadcast detenido.")
            break

# --- Funciones de Control ---
async def broadcast_message(message: dict):
    if g_state["clients"]:
        # Usar list() para evitar "RuntimeError: Set changed size during iteration"
        await asyncio.gather(*[ws.send_json(message) for ws in list(g_state["clients"]) if not ws.closed])

async def stream_subprocess_logs(process):
    while process.returncode is None:
        line = await process.stdout.readline()
        if not line: break
        await broadcast_message({"type": "training_log", "data": line.decode('utf-8', errors='ignore').strip()})
    await process.wait()
    await broadcast_message({"type": "status", "data": "Proceso de entrenamiento finalizado."})
    g_state["training_process"] = None

def get_checkpoints():
    train_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "**", "*.pth"), recursive=True)
    train_files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.relpath(f, cfg.CHECKPOINT_DIR) for f in train_files]

async def start_simulation(args):
    if g_state["simulation_tasks"]:
        logging.warning("La simulación ya está en curso. Deteniéndola primero.")
        stop_simulation()

    logging.info("Iniciando simulación...")
    try:
        model_path = args.get('model_path')
        
        # Cargar el modelo
        g_state["sim_model"] = load_model(model_path)
        g_state["sim_model"].to(cfg.DEVICE)
        g_state["sim_model"] = torch.compile(g_state["sim_model"])
        
        # Inicializar el motor de simulación
        # d_vector para Aetheria_Motor debe ser el número total de canales (real + imag)
        g_state["sim_motor"] = Aetheria_Motor(size=cfg.GRID_SIZE, d_vector=cfg.D_STATE * 2, operator_model=g_state["sim_model"])
        g_state["sim_motor"].state._reset_state_complex_noise()
        
        g_state["sim_latest_psi"] = g_state["sim_motor"].state.psi.clone()
        g_state["sim_previous_psi"] = g_state["sim_latest_psi"].clone()
        g_state["sim_step_count"] = 0
        g_state["sim_density_history"].clear()
        g_state["sim_spacetime_buffer"].clear()

        # Iniciar tareas de simulación y broadcast
        sim_loop_task = asyncio.create_task(simulation_loop())
        broad_loop_task = asyncio.create_task(broadcast_loop())
        g_state["simulation_tasks"] = [sim_loop_task, broad_loop_task]

        # Enviar configuración de simulación a la UI
        sim_config_info = {
            "model_path": model_path if model_path else "Default/Random",
            "grid_size": cfg.GRID_SIZE,
            "d_state": cfg.D_STATE,
            "model_architecture": cfg.MODEL_ARCHITECTURE,
            "device": str(cfg.DEVICE)
        }
        await broadcast_message({"type": "sim_status", "status": "running", "config": sim_config_info})
        logging.info("Simulación iniciada con éxito.")

    except Exception as e:
        logging.error(f"Error al iniciar la simulación: {e}", exc_info=True)
        await broadcast_message({"type": "error", "data": f"Error al iniciar la simulación: {e}"})

def stop_simulation():
    logging.info("Deteniendo simulación...")
    for task in g_state["simulation_tasks"]:
        task.cancel()
    g_state["simulation_tasks"] = []
    g_state["sim_motor"] = None
    g_state["sim_model"] = None
    logging.info("Simulación detenida.")
    asyncio.create_task(broadcast_message({"type": "sim_status", "status": "stopped"}))

# --- Manejador de WebSocket Unificado ---
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    g_state["clients"].add(ws)
    g_state["sim_connected_clients"][ws] = {"viewport": DEFAULT_VIEWPORT.copy()}
    logging.info(f"Cliente conectado: {request.remote}. Clientes totales: {len(g_state['clients'])}")

    # Enviar estado inicial
    initial_sim_config = { # Configuración actual de la simulación si está activa
        "model_path": "N/A",
        "grid_size": cfg.GRID_SIZE,
        "d_state": cfg.D_STATE,
        "model_architecture": cfg.MODEL_ARCHITECTURE,
        "device": str(cfg.DEVICE)
    } if g_state["sim_motor"] else {}

    await ws.send_json({
        "type": "init",
        "checkpoints": get_checkpoints(),
        "is_training": g_state["training_process"] is not None,
        "is_simulating": bool(g_state["simulation_tasks"]),
        "sim_config": initial_sim_config
    })

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    scope = data.get("scope", "lab")
                    command = data.get("command")
                    args = data.get("args", {})

                    if scope == "lab":
                        if command == "start_training":
                            if g_state["training_process"]:
                                await ws.send_json({"type": "error", "data": "El entrenamiento ya está en curso."})
                                continue
                            logging.info(f"Recibida orden de ENTRENAR con args: {args}")
                            
                            # Construir el comando para train.py
                            cmd = [
                                sys.executable, 
                                os.path.join(PROJECT_ROOT, "scripts", "train.py"),
                                "--name", args.get("name", cfg.EXPERIMENT_NAME),
                                "--model", args.get("model", "unet"),
                                "--lr", str(args.get("lr", cfg.LR_RATE_M)),
                                "--episodes", str(args.get("episodes", cfg.EPISODES_TO_ADD)),
                                "--hidden_channels", str(args.get("hidden_channels", cfg.HIDDEN_CHANNELS))
                            ]
                            
                            g_state["training_process"] = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                            await broadcast_message({"type": "status", "data": "Iniciando entrenamiento..."})
                            asyncio.create_task(stream_subprocess_logs(g_state["training_process"]))
                        
                        elif command == "stop_training":
                            if g_state["training_process"]:
                                g_state["training_process"].terminate()
                                await broadcast_message({"type": "status", "data": "Entrenamiento detenido por el usuario."})
                        
                        elif command == "refresh_checkpoints":
                            await ws.send_json({"type": "checkpoints", "data": get_checkpoints()})

                    elif scope == "sim":
                        if command == "start": await start_simulation(args)
                        elif command == "stop": stop_simulation()
                        elif command == "pause": g_state["sim_is_paused"] = True
                        elif command == "resume": g_state["sim_is_paused"] = False
                        elif command == "reset":
                            if g_state["sim_motor"]:
                                g_state["sim_motor"].state._reset_state_complex_noise()
                                g_state["sim_step_count"] = 0
                                g_state["sim_density_history"].clear()
                                g_state["sim_spacetime_buffer"].clear()
                                g_state["sim_cube_buffer"].clear() # <-- ¡NUEVO!
                                g_state["sim_latest_psi"] = g_state["sim_motor"].state.psi.clone()
                                g_state["sim_previous_psi"] = g_state["sim_latest_psi"].clone()
                                for client_ws in g_state["sim_connected_clients"]:
                                    g_state["sim_connected_clients"][client_ws]['viewport'] = DEFAULT_VIEWPORT.copy()
                                logging.info("Simulación reseteada.")
                        elif command == "set_viz": g_state["sim_viz_type"] = args.get('type', 'density')
                        elif command == "set_viewport":
                            if ws in g_state["sim_connected_clients"]:
                                g_state["sim_connected_clients"][ws]['viewport'] = args.get('viewport', DEFAULT_VIEWPORT)

                except Exception as e:
                    logging.error(f"Error procesando comando: {e}", exc_info=True)
            elif msg.type == web.WSMsgType.ERROR:
                logging.error(f"Error de WebSocket: {ws.exception()}", exc_info=True)
    finally:
        logging.info(f"Cliente desconectado: {request.remote}")
        g_state["clients"].remove(ws)
        if ws in g_state["sim_connected_clients"]:
            del g_state["sim_connected_clients"][ws]
    return ws

# --- Configuración y Punto de Entrada ---
def setup_app():
    app = web.Application()
    app.router.add_get('/', lambda r: web.FileResponse(os.path.join(PROJECT_ROOT, "web", "index.html")))
    app.router.add_get('/ws', websocket_handler)
    logging.info(f"Servidor de Aetheria iniciado en http://{cfg.LAB_SERVER_HOST}:{cfg.LAB_SERVER_PORT}")
    return app

if __name__ == "__main__":
    try:
        app = setup_app()
        web.run_app(app, host=cfg.LAB_SERVER_HOST, port=cfg.LAB_SERVER_PORT)
    except KeyboardInterrupt:
        logging.info("Apagado solicitado por el usuario (Ctrl+C).")
        # Cancelar todas las tareas pendientes al salir
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
