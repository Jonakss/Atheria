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
import websockets
import json
import base64
from io import BytesIO
from PIL import Image

# (No hay importaciones de 'lightning.app' aquí)

# ¡Importaciones relativas!
from . import config as cfg
from .qca_engine import QCA_State, Aetheria_Motor # Motor y Estado
from .visualization import (
    downscale_frame, get_density_frame_gpu, get_channel_frame_gpu,
    get_state_magnitude_frame_gpu, get_state_phase_frame_gpu,
    get_state_change_magnitude_frame_gpu
)
from .utils import load_qca_state, save_qca_state

# --- Selector de Modelo (Ley M) ---
# Opción 1: El MLP 1x1 original (Rápido, pero "míope")
# from .qca_operator_mlp import QCA_Operator_MLP as ActiveModel

# Opción 2: La U-Net (Más lenta, pero con "conciencia regional")
from .qca_operator_unet import QCA_Operator_UNet as ActiveModel
# -----------------------------------------------


# --- ¡NUEVO! Clase de Estado Global ---
class GlobalState:
    def __init__(self):
        self.viz_type = cfg.REAL_TIME_VIZ_TYPE
        self.pause_event = asyncio.Event()
        self.pause_event.set() # .set() significa "no pausado" (continúa)
        self.reset_event = asyncio.Event()
        
g_state = GlobalState()
# ------------------------------------

# --- Globales del Servidor WebSocket ---
g_data_queue = asyncio.Queue(maxsize=10)
g_clients = set()

# --- Funciones de Ayuda del Servidor ---

def numpy_to_base64_png(frame_numpy):
    """Convierte un frame de numpy (H, W, 3) a un string Base64 PNG."""
    try:
        img = Image.fromarray(frame_numpy, 'RGB')
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error convirtiendo frame a Base64: {e}")
        return None

async def handle_client_commands(websocket):
    """Escucha los comandos de un cliente específico."""
    print(f"🔌 Cliente conectado: {websocket.remote_address}")
    g_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get("command")
                
                if command == "set_viz":
                    g_state.viz_type = data.get("type", cfg.REAL_TIME_VIZ_TYPE)
                    print(f"🖥️  Cliente cambió el tipo de visualización a: {g_state.viz_type}")
                
                elif command == "pause":
                    g_state.pause_event.clear() # .clear() significa "pausar"
                    print("⏸️  Simulación pausada por el cliente.")
                
                elif command == "resume":
                    g_state.pause_event.set() # .set() significa "continuar"
                    print("▶️  Simulación reanudada por el cliente.")
                
                elif command == "reset":
                    g_state.reset_event.set() # Activa el flag de reseteo
                    print("🔄  Cliente solicitó reseteo de la simulación.")
            
            except json.JSONDecodeError:
                print(f"Error: Mensaje no válido recibido de {websocket.remote_address}")
            except Exception as e:
                print(f"Error procesando comando: {e}")
    finally:
        print(f"🔌 Cliente desconectado: {websocket.remote_address}")
        g_clients.remove(websocket)

async def broadcast_data_loop():
    """Espera datos de la simulación y los envía a todos los clientes."""
    while True:
        data_package = await g_data_queue.get()
        if not g_clients:
            g_data_queue.task_done()
            continue

        step = data_package.get('step', 0)
        current_viz_type = data_package.get('viz_type')
        frame_to_send = data_package.get('frame')

        if frame_to_send is None:
            g_data_queue.task_done()
            continue
        
        frame_b64 = await asyncio.to_thread(numpy_to_base64_png, frame_to_send)
        
        if not frame_b64:
            g_data_queue.task_done()
            continue
            
        payload = json.dumps({
            'step': step,
            'frame_type': current_viz_type,
            'image_data': frame_b64
        })

        clients_to_remove = []
        for client in list(g_clients): # Iterar sobre una copia para modificar el original
            try:
                await client.send(payload)
            except websockets.exceptions.ConnectionClosed:
                print(f"🔌 Cliente {client.remote_address} se ha desconectado.")
                clients_to_remove.append(client)
            except Exception as e:
                print(f"⚠️  Error inesperado al enviar a cliente {client.remote_address}: {type(e).__name__}: {e}")
                clients_to_remove.append(client)
        
        for client in clients_to_remove:
            g_clients.remove(client)
        
        g_data_queue.task_done()

async def run_simulation_loop(motor, start_step):
    """Bucle principal de simulación que obedece al estado global."""
    print(f"\n🎬 Iniciando simulación infinita en {motor.size}x{motor.size}...")
    
    t = start_step

    # Arreglo de fuga de memoria: crea prev_state UNA VEZ
    prev_state = QCA_State(motor.state.size, motor.state.d_state)
    prev_state.x_real.data = motor.state.x_real.data.clone().to(cfg.DEVICE)
    prev_state.x_imag.data = motor.state.x_imag.data.clone().to(cfg.DEVICE)

    with torch.no_grad():
        while True:
            # --- 1. Comprobar comandos de control ---
            
            await g_state.pause_event.wait() # Si está "pausado", espera aquí
            
            if g_state.reset_event.is_set():
                print("🔄  Reseteando estado de la simulación...")
                if cfg.INITIAL_STATE_MODE_INFERENCE == 'complex_noise':
                    motor.state._reset_state_complex_noise()
                else:
                    motor.state._reset_state_random()
                
                prev_state.x_real.data = motor.state.x_real.data.clone().to(cfg.DEVICE)
                prev_state.x_imag.data = motor.state.x_imag.data.clone().to(cfg.DEVICE)
                t = 0
                g_state.reset_event.clear() # Baja el flag de reseteo

            # 2. Evolucionar el estado
            await asyncio.to_thread(motor.evolve_step)
            current_state = motor.state

            # 3. Capturar y enviar frames
            if (t + 1) % cfg.REAL_TIME_VIZ_INTERVAL == 0:
                try:
                    current_viz_type = g_state.viz_type # Leer el tipo de la UI
                    
                    frame = None
                    if current_viz_type == 'density':
                        frame = await asyncio.to_thread(get_density_frame_gpu, current_state)
                    elif current_viz_type == 'channels':
                        frame = await asyncio.to_thread(get_channel_frame_gpu, current_state, num_channels=min(3, cfg.D_STATE))
                    elif current_viz_type == 'magnitude':
                        frame = await asyncio.to_thread(get_state_magnitude_frame_gpu, current_state)
                    elif current_viz_type == 'phase':
                        frame = await asyncio.to_thread(get_state_phase_frame_gpu, current_state)
                    elif current_viz_type == 'change':
                        frame = await asyncio.to_thread(get_state_change_magnitude_frame_gpu, current_state, prev_state)

                    if frame is not None:
                        if cfg.REAL_TIME_VIZ_DOWNSCALE > 1:
                            frame = await asyncio.to_thread(downscale_frame, frame, cfg.REAL_TIME_VIZ_DOWNSCALE)
                        
                        data_package = {
                            'step': t + 1,
                            'viz_type': current_viz_type,
                            'frame': frame
                        }
                        
                        try:
                            g_data_queue.put_nowait(data_package)
                        except asyncio.QueueFull:
                            print(f"⚠️  Cola de datos llena. Descartando frame del paso {t+1}.")
                
                except Exception as e:
                     print(f"⚠️  Error al generar frame en paso {t+1}: {e}")

            # 4. Actualizar prev_state para la próxima iteración
            prev_state.x_real.data = current_state.x_real.data.clone().to(cfg.DEVICE)
            prev_state.x_imag.data = current_state.x_imag.data.clone().to(cfg.DEVICE)

            # 5. Guardar Checkpoint
            if cfg.LARGE_SIM_CHECKPOINT_INTERVAL and (t + 1) % cfg.LARGE_SIM_CHECKPOINT_INTERVAL == 0:
                await asyncio.to_thread(
                    save_qca_state, 
                    motor, t + 1, 
                    cfg.LARGE_SIM_CHECKPOINT_DIR
                )

            # 6. Imprimir progreso
            if (t + 1) % 100 == 0: 
                print(f"📈 Progreso Simulación: Paso {t+1}. Clientes: {len(g_clients)}. Cola: {g_data_queue.qsize()}")
            
            t += 1
            await asyncio.sleep(0.001)

# --- Función Principal del Servidor ---
async def run_server_pipeline(M_FILENAME: str | None):
    """
    Ejecuta la FASE 7: Inicia el servidor de simulación grande.
    Esta versión es agnóstica a Lightning.
    """
    print("\n" + "="*60)
    print(">>> INICIANDO FASE DE SIMULACIÓN GRANDE (FASE 7) COMO SERVIDOR <<<")
    print(f"Modelo Activo: {ActiveModel.__name__}")
    print(f"Modo: Local/Agnóstico (Controlado por WebSocket)")
    print("="*60)

    # --- 7.1: Configuración Síncrona ---
    
    operator_model_inference = ActiveModel(
        d_state=cfg.D_STATE,
        hidden_channels=cfg.HIDDEN_CHANNELS
    )

    if cfg.DEVICE.type == 'cuda':
        try:
            print("Aplicando torch.compile() al modelo de inferencia...")
            operator_model_inference = torch.compile(operator_model_inference, mode="reduce-overhead")
            print("¡torch.compile() aplicado exitosamente!")
        except Exception as e:
            print(f"Advertencia: torch.compile() falló: {e}")
    else:
        print("INFO: Omitiendo torch.compile() en CPU.")

    large_scale_motor = Aetheria_Motor(cfg.GRID_SIZE_INFERENCE, cfg.D_STATE, operator_model_inference)

    model_id = ActiveModel.__name__
    if not M_FILENAME:
         model_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, f"{model_id}_G{cfg.GRID_SIZE_TRAINING}_Eps*_FINAL.pth"))
         if not model_files: 
             model_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, f"PEF_Deep_v3_G{cfg.GRID_SIZE_TRAINING}_Eps*_FINAL.pth"))
         M_FILENAME = max(model_files, key=os.path.getctime, default=None) if model_files else None

    if M_FILENAME and os.path.exists(M_FILENAME):
        print(f"📦 Cargando pesos desde: {M_FILENAME}")
        try:
            model_state_dict = torch.load(M_FILENAME, map_location=cfg.DEVICE)
            target_model = large_scale_motor.operator
            if isinstance(target_model, nn.DataParallel):
                target_model = target_model.module
            if hasattr(target_model, '_orig_mod'):
                 target_model = target_model._orig_mod
            is_dataparallel_saved = next(iter(model_state_dict)).startswith('module.')
            if is_dataparallel_saved:
                new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
                target_model.load_state_dict(new_state_dict)
            else:
                target_model.load_state_dict(model_state_dict)
            large_scale_motor.operator.eval()
            print(f"✅ Pesos del modelo cargados exitosamente.")
        except Exception as e:
            print(f"❌ Error cargando pesos del modelo '{M_FILENAME}': {e}")
            print("⚠️  La simulación se ejecutará con pesos aleatorios.")
    else:
        print(f"❌ No se encontró archivo de modelo entrenado. Usando pesos aleatorios.")
    
    start_step = 0
    if cfg.LOAD_STATE_CHECKPOINT_INFERENCE:
        latest_checkpoint_filepath = cfg.STATE_CHECKPOINT_PATH_INFERENCE
        if not latest_checkpoint_filepath or not os.path.exists(latest_checkpoint_filepath):
             checkpoint_files = [f for f in os.listdir(cfg.LARGE_SIM_CHECKPOINT_DIR) if f.startswith("large_sim_state_step_") and f.endswith(".pth")]
             if checkpoint_files:
                 def extract_step(f):
                     match = re.search(r"large_sim_state_step_(\d+)\.pth", f)
                     return int(match.group(1)) if match else 0
                 checkpoint_files.sort(key=extract_step)
                 latest_checkpoint_filepath = os.path.join(cfg.LARGE_SIM_CHECKPOINT_DIR, checkpoint_files[-1])
        
        if latest_checkpoint_filepath and os.path.exists(latest_checkpoint_filepath):
            print(f"Cargando estado desde: {latest_checkpoint_filepath}")
            loaded_step = load_qca_state(large_scale_motor, latest_checkpoint_filepath)
            if loaded_step != -1:
                start_step = loaded_step
                print(f"Simulación reanudada desde el paso {start_step}.")
    
    if start_step == 0:
        print(f"\nIniciando nueva simulación con modo: '{cfg.INITIAL_STATE_MODE_INFERENCE}'.")
        
        # --- ¡¡AQUÍ ESTÁ LA CORRECCIÓN!! ---
        # El nombre de la variable es 'large_scale_motor', no 'motor'
        if cfg.INITIAL_STATE_MODE_INFERENCE == 'complex_noise':
            large_scale_motor.state._reset_state_complex_noise()
        else:
            large_scale_motor.state._reset_state_random()

    print("✅ Configuración de simulación completada.")

    # --- 7.2: Iniciar Tareas Asíncronas ---
    print(f"\n--- 🚀 Iniciando Tareas Asíncronas ---")
    
    broadcast_task = asyncio.create_task(broadcast_data_loop())
    simulation_task = asyncio.create_task(run_simulation_loop(large_scale_motor, start_step))

    print(f"--- ✅ Iniciando Servidor WebSocket en ws://{cfg.WEBSOCKET_HOST}:{cfg.WEBSOCKET_PORT} ---")
    
    async with websockets.serve(handle_client_commands, cfg.WEBSOCKET_HOST, cfg.WEBSOCKET_PORT):
        print("✅ Servidor iniciado. Simulación y broadcast corriendo en segundo plano.")
        await asyncio.gather(broadcast_task, simulation_task)