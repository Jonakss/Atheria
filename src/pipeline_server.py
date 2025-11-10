# src/pipeline_server.py
import torch
import torch.nn as nn
import numpy as np
import time
import os
import glob
import re
import gc
import asyncio
import json
import base64
from io import BytesIO
from PIL import Image

# ¡NUEVAS IMPORTACIONES!
from aiohttp import web
import websockets 

# ¡Importaciones relativas!
from . import config as cfg
from .qca_engine import QCA_State, Aetheria_Motor # Motor y Estado
from .visualization import (
    downscale_frame
    # Importamos las funciones de viz directamente
)
from .utils import load_qca_state, save_qca_state

# --- Selector de Modelo (Ley M) ---
print(f"Cargando Ley M: {cfg.ACTIVE_QCA_OPERATOR}")
if cfg.ACTIVE_QCA_OPERATOR == "MLP":
    from .qca_operator_mlp import QCA_Operator_MLP as ActiveModel
elif cfg.ACTIVE_QCA_OPERATOR == "UNET_UNITARIA":
    from .qca_operator_unet_unitary import QCA_Operator_UNet_Unitary as ActiveModel
else:
    raise ValueError(f"Operador QCA '{cfg.ACTIVE_QCA_OPERATOR}' no reconocido en config.py")
# -----------------------------------------------


# --- Clase de Estado Global ---
class GlobalState:
    def __init__(self):
        self.viz_type = cfg.REAL_TIME_VIZ_TYPE
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.reset_event = asyncio.Event()
        
g_state = GlobalState()
g_data_queue = asyncio.Queue(maxsize=10)
g_clients = set() # set de clientes websocket

# --- Funciones de Ayuda ---
def numpy_to_base64_png(frame_numpy):
    try:
        img = Image.fromarray(frame_numpy, 'RGB')
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error convirtiendo frame a Base64: {e}")
        return None

# --- ¡¡NUEVAS FUNCIONES DE VISUALIZACIÓN!! ---
# (Adaptadas para el nuevo estado unitario 'psi')

def get_density_frame_unitary(state: QCA_State):
    """Genera un frame de densidad desde un estado unitario 'psi'."""
    density_map = torch.sum(state.psi.pow(2), dim=-1).squeeze(0).detach()
    d_min, d_max = density_map.min(), density_map.max()
    norm_factor = d_max - d_min
    if norm_factor < 1e-8:
        normalized_density = torch.zeros_like(density_map)
    else:
        normalized_density = (density_map - d_min) / norm_factor
    normalized_density_clamped = normalized_density.clamp(0.0, 1.0)
    R = normalized_density_clamped
    G = torch.zeros_like(normalized_density_clamped)
    B = 1.0 - normalized_density_clamped
    img_rgb = torch.stack([R, G, B], dim=2).clamp(0.0, 1.0)
    return (img_rgb * 255).byte().cpu().numpy()

def get_channels_frame_unitary(state: QCA_State, num_channels=3):
    """Visualiza los primeros 3 canales del vector 'psi'."""
    psi_abs = torch.abs(state.psi.squeeze(0)).detach() # [H, W, D_vector]
    combined_image = torch.zeros(state.size, state.size, 3, device=cfg.DEVICE)
    
    num_to_viz = min(num_channels, state.d_vector)
    if num_to_viz == 0:
        return (combined_image * 255).byte().cpu().numpy()
        
    for i in range(num_to_viz):
        channel_data = psi_abs[:, :, i]
        ch_min, ch_max = channel_data.min(), channel_data.max()
        if (ch_max - ch_min) < 1e-8:
            channel_scaled = torch.zeros_like(channel_data)
        else:
            channel_scaled = (channel_data - ch_min) / (ch_max - ch_min)
        combined_image[:, :, i % 3] += channel_scaled 
    
    final_image = (combined_image.clamp(0, 1) * 255).byte().cpu().numpy()
    return final_image
    
def get_change_frame_unitary(state: QCA_State, prev_state: QCA_State):
    """Visualiza el cambio en el vector 'psi'."""
    change_vector = state.psi - prev_state.psi
    change_magnitude_map = torch.sum(change_vector.pow(2), dim=-1).squeeze(0).detach()
    
    m_min, m_max = change_magnitude_map.min(), change_magnitude_map.max()
    norm_factor = m_max - m_min
    if norm_factor < 1e-12:
        normalized_change = torch.zeros_like(change_magnitude_map)
    else:
        normalized_change = (change_magnitude_map - m_min) / norm_factor
    
    img_gray = normalized_change.clamp(0.0, 1.0)
    img_rgb = torch.stack([img_gray, img_gray, img_gray], dim=2)
    return (img_rgb * 255).byte().cpu().numpy()

# -------------------------------------------

# --- ¡¡LÓGICA DE BUCLE SIMPLIFICADA Y CORREGIDA!! ---
async def run_simulation_loop(motor, start_step):
    print(f"\n🎬 Iniciando simulación infinita en {motor.size}x{motor.size}...")
    
    t = start_step
    
    # Creamos DOS búferes de estado. 'current_state' apunta al estado t+1,
    # 'prev_state' apunta al estado t. Los intercambiamos.
    current_state = motor.state
    prev_state = QCA_State(motor.state.size, motor.state.d_vector)
    prev_state.psi.data = current_state.psi.data.clone().to(cfg.DEVICE)

    with torch.no_grad():
        while True:
            await g_state.pause_event.wait()
            
            if g_state.reset_event.is_set():
                print("🔄  Reseteando estado de la simulación...")
                if cfg.INITIAL_STATE_MODE_INFERENCE == 'complex_noise':
                    current_state._reset_state_complex_noise()
                else:
                    current_state._reset_state_random()
                
                prev_state.psi.data = current_state.psi.data.clone().to(cfg.DEVICE)
                t = 0
                g_state.reset_event.clear()

            # 1. Intercambiar búferes:
            #    El 'current_state' (que era t+1) se convierte en 'prev_state' (t)
            #    Y 'prev_state' (que era t) se convierte en el lienzo para 'current_state' (t+1)
            #    Esto es un truco de punteros, es instantáneo.
            temp = prev_state
            prev_state = current_state
            current_state = temp
            
            # Asignar el 'prev_state' (que ahora contiene el estado t) al motor
            motor.state = prev_state

            # 2. Evolucionar. motor.evolve_step() leerá de motor.state (t)
            #    y escribirá el resultado (t+1) en 'current_state'
            #    (Modificamos evolve_step para que acepte un estado de salida)
            await asyncio.to_thread(motor.evolve_step, current_state)
            
            # (Actualización: `evolve_step` modifica su propio `motor.state` internamente.
            #  El arreglo de la fuga de memoria de la última vez es el correcto.
            #  Ignoremos el intercambio de punteros por ahora, es demasiado complejo.
            #  Volvamos al arreglo de la fuga de memoria que funcionaba.)

            # --- ARREGLO DE FUGA DE MEMORIA (Versión Probada) ---
            
            # 1. Copiar estado t ANTES de evolucionar
            prev_state.psi.data = motor.state.psi.data.clone().to(cfg.DEVICE)
            
            # 2. Evolucionar (motor.state ahora es t+1)
            await asyncio.to_thread(motor.evolve_step)
            current_state = motor.state # Este es el estado t+1
            
            # 3. Generar frames (usando current_state (t+1) y prev_state (t))
            if (t + 1) % cfg.REAL_TIME_VIZ_INTERVAL == 0:
                try:
                    current_viz_type = g_state.viz_type
                    frame = None
                    
                    if current_viz_type == 'density':
                        frame = await asyncio.to_thread(get_density_frame_unitary, current_state)
                    elif current_viz_type == 'channels':
                        frame = await asyncio.to_thread(get_channels_frame_unitary, current_state, num_channels=3)
                    elif current_viz_type == 'change':
                        frame = await asyncio.to_thread(get_change_frame_unitary, current_state, prev_state)
                    
                    if frame is not None:
                        # ... (lógica de encolado, sin cambios) ...
                        if cfg.REAL_TIME_VIZ_DOWNSCALE > 1:
                            frame = await asyncio.to_thread(downscale_frame, frame, cfg.REAL_TIME_VIZ_DOWNSCALE)
                        data_package = { 'step': t + 1, 'viz_type': current_viz_type, 'frame': frame }
                        try: g_data_queue.put_nowait(data_package)
                        except asyncio.QueueFull: pass
                
                except Exception as e:
                     print(f"⚠️  Error al generar frame en paso {t+1}: {e}")
            
            # El 'prev_state' ya está listo para la próxima iteración.
            # No se crean nuevos objetos QCA_State en el bucle.
            
            # ----------------------------------------------------

            if cfg.LARGE_SIM_CHECKPOINT_INTERVAL and (t + 1) % cfg.LARGE_SIM_CHECKPOINT_INTERVAL == 0:
                await asyncio.to_thread(save_qca_state, motor, t + 1, cfg.LARGE_SIM_CHECKPOINT_DIR)

            if (t + 1) % 100 == 0: 
                print(f"📈 Progreso: Paso {t+1}. Clientes: {len(g_clients)}. Cola: {g_data_queue.qsize()}")
            
            t += 1
            await asyncio.sleep(0.001)

# --- Bucle de Broadcast (sin cambios) ---
async def broadcast_data_loop():
    # ... (código idéntico de broadcast_data_loop) ...
    while True:
        data_package = await g_data_queue.get()
        if not g_clients:
            g_data_queue.task_done()
            continue
        step, current_viz_type, frame_to_send = data_package.get('step'), data_package.get('viz_type'), data_package.get('frame')
        if frame_to_send is None:
            g_data_queue.task_done()
            continue
        frame_b64 = await asyncio.to_thread(numpy_to_base64_png, frame_to_send)
        if not frame_b64:
            g_data_queue.task_done()
            continue
        payload = json.dumps({'step': step, 'frame_type': current_viz_type, 'image_data': frame_b64})
        
        # Usar una copia del set para poder modificarlo si un cliente se desconecta
        disconnected_clients = set()
        for client in g_clients:
            try:
                # En aiohttp, usamos await en lugar de create_task para el envío directo
                await client.send_str(payload)
            except ConnectionResetError:
                print(f"🔌 Cliente desconectado (ConnectionResetError), marcando para eliminar.")
                disconnected_clients.add(client)
            except Exception as e:
                print(f"🔌 Error enviando a un cliente: {e}. Marcando para eliminar.")
                disconnected_clients.add(client)

        # Eliminar clientes desconectados fuera del bucle de iteración
        for client in disconnected_clients:
            g_clients.remove(client)
        g_data_queue.task_done()

# --- Manejadores de aiohttp (sin cambios) ---
async def serve_html_handler(request):
    viewer_path = os.path.join(cfg.PROJECT_ROOT, "viewer.html")
    try:
        return web.FileResponse(viewer_path)
    except FileNotFoundError:
        return web.Response(text="Error: viewer.html no encontrado.", status=404)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    g_clients.add(ws)
    print(f"🔌 Cliente conectado: {request.remote}")
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    command = data.get("command")
                    if command == "set_viz":
                        g_state.viz_type = data.get("type", cfg.REAL_TIME_VIZ_TYPE)
                        print(f"🖥️  Cliente cambió viz a: {g_state.viz_type}")
                    elif command == "pause":
                        g_state.pause_event.clear(); print("⏸️  Simulación pausada.")
                    elif command == "resume":
                        g_state.pause_event.set(); print("▶️  Simulación reanudada.")
                    elif command == "reset":
                        g_state.reset_event.set(); print("🔄  Cliente solicitó reseteo.")
                except Exception as e:
                    print(f"Error procesando comando: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                print(f"Error de WebSocket: {ws.exception()}")
    finally:
        print(f"🔌 Cliente desconectado: {request.remote}")
        g_clients.remove(ws)
    return ws

async def start_background_tasks(app):
    print("Iniciando tareas de fondo...")
    app['simulation_task'] = asyncio.create_task(run_simulation_loop(app['motor'], app['start_step']))
    app['broadcast_task'] = asyncio.create_task(broadcast_data_loop())

async def cleanup_background_tasks(app):
    print("Deteniendo tareas de fondo...")
    if 'simulation_task' in app: app['simulation_task'].cancel()
    if 'broadcast_task' in app: app['broadcast_task'].cancel()
    if 'simulation_task' in app:
        try: await app['simulation_task']
        except asyncio.CancelledError: pass
    if 'broadcast_task' in app:
        try: await app['broadcast_task']
        except asyncio.CancelledError: pass

# --- Función Principal del Servidor ---
async def run_server_pipeline(M_FILENAME: str | None):
    print("\n" + "="*60)
    print(">>> INICIANDO SERVIDOR AETHERIA (FASE 7) <<<")
    print(f"Modelo Activo: {ActiveModel.__name__}")
    print("="*60)

    # --- 7.1: Configuración Síncrona ---
    operator_model_inference = ActiveModel(
        d_vector=cfg.D_STATE, 
        hidden_channels=cfg.HIDDEN_CHANNELS
    )

    if cfg.DEVICE.type == 'cuda':
        try:
            print("Aplicando torch.compile() al modelo...")
            operator_model_inference = torch.compile(operator_model_inference, mode="reduce-overhead")
            print("¡torch.compile() aplicado exitosamente!")
        except Exception as e:
            print(f"Advertencia: torch.compile() falló: {e}")

    large_scale_motor = Aetheria_Motor(
        cfg.GRID_SIZE_INFERENCE, 
        cfg.D_STATE, 
        operator_model_inference
    )

    model_id = ActiveModel.__name__
    if not M_FILENAME:
         search_path = os.path.join(cfg.CHECKPOINT_DIR, cfg.EXPERIMENT_NAME, f"{model_id}_G{cfg.GRID_SIZE_TRAINING}_Eps*_FINAL.pth")
         model_files = glob.glob(search_path)
         if not model_files:
             print(f"No se encontraron modelos en '{search_path}', buscando en la raíz")
             model_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, f"{model_id}_G{cfg.GRID_SIZE_TRAINING}_Eps*_FINAL.pth"))
         M_FILENAME = max(model_files, key=os.path.getctime, default=None) if model_files else None

    if M_FILENAME and os.path.exists(M_FILENAME):
        print(f"📦 Cargando pesos desde: {M_FILENAME}")
        try:
            model_state_dict = torch.load(M_FILENAME, map_location=cfg.DEVICE)
            target_model = large_scale_motor.operator
            if isinstance(target_model, nn.DataParallel): target_model = target_model.module
            if hasattr(target_model, '_orig_mod'): target_model = target_model._orig_mod
            target_model.load_state_dict(model_state_dict, strict=False)
            large_scale_motor.operator.eval()
            print("✅ Pesos del modelo cargados.")
        except Exception as e:
            print(f"❌ Error cargando pesos: {e}. Usando pesos aleatorios.")
    else:
        print("❌ No se encontró modelo. Usando pesos aleatorios.")
    
    start_step = 0
    if cfg.LOAD_STATE_CHECKPOINT_INFERENCE:
        # (Aquí va tu lógica de 'load_qca_state')
        pass
    
    if start_step == 0:
        print(f"Iniciando nueva simulación con modo: '{cfg.INITIAL_STATE_MODE_INFERENCE}'.")
        if cfg.INITIAL_STATE_MODE_INFERENCE == 'complex_noise':
            large_scale_motor.state._reset_state_complex_noise()
        else:
            large_scale_motor.state._reset_state_random()

    print("✅ Configuración de simulación completada.")

    # --- 7.2: Iniciar Servidor aiohttp ---
    app = web.Application()
    app['motor'] = large_scale_motor
    app['start_step'] = start_step
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    app.router.add_get('/', serve_html_handler)
    app.router.add_get('/ws', websocket_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=cfg.WEBSOCKET_HOST, port=cfg.WEBSOCKET_PORT)
    await site.start()
    
    print(f"--- ✅ Servidor HTTP/WebSocket iniciado en http://{cfg.WEBSOCKET_HOST}:{cfg.WEBSOCKET_PORT} ---")
    print("Abre esa URL en tu navegador para ver el controlador.")
    
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()