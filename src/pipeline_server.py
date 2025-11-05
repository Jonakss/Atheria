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

# ¡Importaciones relativas!
from . import config as cfg
from .qca_engine import QCA_State, Aetheria_Motor # Motor y Estado
from .visualization import (
    downscale_frame, get_density_frame_gpu, get_channel_frame_gpu,
    get_state_magnitude_frame_gpu, get_state_phase_frame_gpu,
    get_state_change_magnitude_frame_gpu
)
from .utils import load_qca_state, save_qca_state

# --- CAMBIO AQUÍ: Selector de Modelo (Ley M) ---
# Asegúrate de que esto coincida con el modelo que entrenaste.

# Opción 1: El MLP 1x1 original (Rápido, pero "míope")
# from .qca_operator_mlp import QCA_Operator_MLP as ActiveModel

# Opción 2: La U-Net (Más lenta, pero con "conciencia regional")
from .qca_operator_unet import QCA_Operator_UNet as ActiveModel
# -----------------------------------------------


# --- Globales del Servidor WebSocket ---
g_data_queue = asyncio.Queue(maxsize=5)
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

async def register_client(websocket):
    """Registra un nuevo cliente y espera a que se desconecte."""
    print(f"🔌 Cliente conectado: {websocket.remote_address}")
    g_clients.add(websocket)
    try:
        await websocket.wait_closed()
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
        
        frame_b64 = await asyncio.to_thread(
            numpy_to_base64_png, 
            data_package.get(cfg.REAL_TIME_VIZ_TYPE)
        )
        
        if not frame_b64:
            g_data_queue.task_done()
            continue
            
        payload = json.dumps({
            'step': step,
            'frame_type': cfg.REAL_TIME_VIZ_TYPE,
            'image_data': frame_b64
        })

        clients_to_send = list(g_clients) 
        for client in clients_to_send:
            if client.open:
                try:
                    await client.send(payload)
                except websockets.ConnectionClosed:
                    pass
        
        if step % 100 == 0:
             print(f"📡 Datos del paso {step} enviados a {len(clients_to_send)} clientes.")
        g_data_queue.task_done()

async def run_simulation_loop(motor, start_step):
    """
    Bucle principal de simulación que se ejecuta de forma asíncrona.
    """
    print(f"\n🎬 Iniciando simulación infinita en {motor.size}x{motor.size}...")
    print(f"📡 Datos se enviarán por socket cada {cfg.REAL_TIME_VIZ_INTERVAL} pasos.")
    print(f"💾 Checkpoints se guardarán cada {cfg.LARGE_SIM_CHECKPOINT_INTERVAL} pasos en '{cfg.LARGE_SIM_CHECKPOINT_DIR}'.")
    
    t = start_step
    prev_state_for_change_viz = None
    
    if cfg.REAL_TIME_VIZ_TYPE == 'change':
        print("Inicializando estado previo para visualización de 'change'...")
        prev_state_for_change_viz = QCA_State(motor.state.size, motor.state.d_state)
        prev_state_for_change_viz.x_real.data = motor.state.x_real.data.clone().to(cfg.DEVICE)
        prev_state_for_change_viz.x_imag.data = motor.state.x_imag.data.clone().to(cfg.DEVICE)

    with torch.no_grad():
        while True:
            current_state_clone_for_change_viz = None
            if prev_state_for_change_viz is not None:
                current_state_clone_for_change_viz = QCA_State(motor.state.size, motor.state.d_state)
                current_state_clone_for_change_viz.x_real.data = motor.state.x_real.data.clone().to(cfg.DEVICE)
                current_state_clone_for_change_viz.x_imag.data = motor.state.x_imag.data.clone().to(cfg.DEVICE)

            await asyncio.to_thread(motor.evolve_step)
            current_state = motor.state

            if (t + 1) % cfg.REAL_TIME_VIZ_INTERVAL == 0:
                try:
                    frame = None
                    if cfg.REAL_TIME_VIZ_TYPE == 'density':
                        frame = await asyncio.to_thread(get_density_frame_gpu, current_state)
                    elif cfg.REAL_TIME_VIZ_TYPE == 'channels':
                        frame = await asyncio.to_thread(get_channel_frame_gpu, current_state, num_channels=min(3, cfg.D_STATE))
                    elif cfg.REAL_TIME_VIZ_TYPE == 'magnitude':
                        frame = await asyncio.to_thread(get_state_magnitude_frame_gpu, current_state)
                    elif cfg.REAL_TIME_VIZ_TYPE == 'phase':
                        frame = await asyncio.to_thread(get_state_phase_frame_gpu, current_state)
                    elif cfg.REAL_TIME_VIZ_TYPE == 'change' and prev_state_for_change_viz:
                        frame = await asyncio.to_thread(get_state_change_magnitude_frame_gpu, current_state, prev_state_for_change_viz)

                    if frame is not None:
                        if cfg.REAL_TIME_VIZ_DOWNSCALE > 1:
                            frame = await asyncio.to_thread(downscale_frame, frame, cfg.REAL_TIME_VIZ_DOWNSCALE)
                        
                        data_package = {'step': t + 1, cfg.REAL_TIME_VIZ_TYPE: frame}
                        
                        try:
                            g_data_queue.put_nowait(data_package)
                        except asyncio.QueueFull:
                            print(f"⚠️  Cola de datos llena. Descartando frame del paso {t+1}.")
                
                except Exception as e:
                     print(f"⚠️  Error al generar frame en paso {t+1}: {e}")

            if current_state_clone_for_change_viz is not None:
                prev_state_for_change_viz = current_state_clone_for_change_viz

            if cfg.LARGE_SIM_CHECKPOINT_INTERVAL and (t + 1) % cfg.LARGE_SIM_CHECKPOINT_INTERVAL == 0:
                await asyncio.to_thread(
                    save_qca_state, 
                    motor, t + 1, 
                    cfg.LARGE_SIM_CHECKPOINT_DIR
                )

            if (t + 1) % 100 == 0: 
                print(f"📈 Progreso Simulación: Paso {t+1}. Clientes: {len(g_clients)}. Cola: {g_data_queue.qsize()}")
            
            t += 1
            await asyncio.sleep(0.001)

# --- Función Principal del Servidor ---

async def run_server_pipeline(M_FILENAME: str | None):
    """
    Ejecuta la FASE 7: Inicia el servidor de simulación grande.
    """
    print("\n" + "="*60)
    print(">>> INICIANDO FASE DE SIMULACIÓN GRANDE (FASE 7) COMO SERVIDOR <<<")
    print(f"Modelo Activo: {ActiveModel.__name__}")
    print("="*60)

    # --- 7.1: Configuración Síncrona (Cargar modelos, etc.) ---
    
    # 1. Crear el modelo
    operator_model_inference = ActiveModel(
        d_state=cfg.D_STATE,
        hidden_channels=cfg.HIDDEN_CHANNELS
    )

    # --- Potenciación: Aplicar torch.compile ---
    if cfg.DEVICE.type == 'cuda':
        try:
            print("Aplicando torch.compile() al modelo de inferencia...")
            operator_model_inference = torch.compile(operator_model_inference, mode="reduce-overhead")
            print("¡torch.compile() aplicado exitosamente!")
        except Exception as e:
            print(f"Advertencia: torch.compile() falló. Se usará el modelo estándar. Error: {e}")

    # 2. Crear el motor con el modelo (genérico)
    large_scale_motor = Aetheria_Motor(cfg.GRID_SIZE_INFERENCE, cfg.D_STATE, operator_model_inference)

    # Buscar modelo si no se pasó (usando el nombre del modelo activo)
    model_id = ActiveModel.__name__
    if not M_FILENAME:
         model_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, f"{model_id}_G{cfg.GRID_SIZE_TRAINING}_Eps*_FINAL.pth"))
         if not model_files: # fallback al nombre antiguo
             model_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, f"PEF_Deep_v3_G{cfg.GRID_SIZE_TRAINING}_Eps*_FINAL.pth"))
         
         M_FILENAME = max(model_files, key=os.path.getctime, default=None) if model_files else None

    # Cargar pesos del modelo
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

    # Cargar estado de simulación (checkpoint)
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
    
    # Iniciar desde cero si no se cargó checkpoint
    if start_step == 0:
        print(f"\nIniciando nueva simulación con modo: '{cfg.INITIAL_STATE_MODE_INFERENCE}'.")
        if cfg.INITIAL_STATE_MODE_INFERENCE == 'random':
            large_scale_motor.state._reset_state_random()
        elif cfg.INITIAL_STATE_MODE_INFERENCE == 'seeded':
            large_scale_motor.state._reset_state_seeded()
        elif cfg.INITIAL_STATE_MODE_INFERENCE == 'complex_noise':
            large_scale_motor.state._reset_state_complex_noise()

    print("✅ Configuración de simulación completada.")

    # --- 7.2: Iniciar Tareas Asíncronas ---
    print(f"\n--- 🚀 Iniciando Tareas Asíncronas ---")
    
    broadcast_task = asyncio.create_task(broadcast_data_loop())
    simulation_task = asyncio.create_task(run_simulation_loop(large_scale_motor, start_step))

    print(f"--- ✅ Iniciando Servidor WebSocket en ws://{cfg.WEBSOCKET_HOST}:{cfg.WEBSOCKET_PORT} ---")
    
    async with websockets.serve(register_client, cfg.WEBSOCKET_HOST, cfg.WEBSOCKET_PORT):
        print("✅ Servidor iniciado. Simulación y broadcast corriendo en segundo plano.")
        await asyncio.gather(broadcast_task, simulation_task)