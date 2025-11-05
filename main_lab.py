# main.py (El nuevo Servidor de Laboratorio)
import asyncio
import os
import sys
import glob
import json
from aiohttp import web
import websockets

# --- Configuración del Path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from src import config as cfg
    from src.pipeline_server import run_server_pipeline # Importamos el *pipeline* de simulación
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos desde 'src': {e}", file=sys.stderr)
    sys.exit(1)

# --- Estado Global del Laboratorio ---
g_state = {
    "training_process": None,
    "simulation_process": None,
    "lab_clients": set(), # Clientes conectados al panel de control
    "sim_clients": set()  # (Esto lo manejará el pipeline_server)
}

# --- Funciones de Ayuda ---
def get_checkpoints():
    """Escanea el disco y devuelve las listas de checkpoints."""
    train_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "**", "*.pth"), recursive=True)
    train_files.sort(key=os.path.getmtime, reverse=True)
    
    sim_files = glob.glob(os.path.join(cfg.LARGE_SIM_CHECKPOINT_DIR, "*.pth"))
    sim_files.sort(key=os.path.getmtime, reverse=True)
    
    return {
        "training_models": [os.path.relpath(f, cfg.CHECKPOINT_DIR) for f in train_files],
        "sim_states": [os.path.basename(f) for f in sim_files]
    }

async def broadcast_lab_message(message: dict):
    """Envía un mensaje a todos los clientes del Laboratorio."""
    if g_state["lab_clients"]:
        payload = json.dumps(message)
        for ws in g_state["lab_clients"]:
            asyncio.create_task(ws.send(payload))

async def stream_subprocess_logs(process, ws):
    """Lee el stdout de un subproceso y lo retransmite al WebSocket."""
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await ws.send(json.dumps({"type": "log", "data": line.decode('utf-8').strip()}))
    
    await process.wait()
    await ws.send(json.dumps({"type": "status", "data": "Proceso de entrenamiento finalizado."}))
    g_state["training_process"] = None

# --- Manejadores del Servidor Web (aiohttp) ---

async def serve_lab_html(request):
    """Sirve la UI principal del Laboratorio (lab.html)"""
    lab_path = os.path.join(cfg.PROJECT_ROOT, "lab.html")
    try:
        return web.FileResponse(lab_path)
    except FileNotFoundError:
        return web.Response(text="Error: lab.html no encontrado.", status=404)

async def serve_viewer_html(request):
    """Sirve la UI del Visor de Simulación (viewer.html)"""
    viewer_path = os.path.join(cfg.PROJECT_ROOT, "viewer.html")
    try:
        return web.FileResponse(viewer_path)
    except FileNotFoundError:
        return web.Response(text="Error: viewer.html no encontrado.", status=404)

async def lab_websocket_handler(request):
    """Maneja los comandos del Laboratorio (Entrenar, Parar, etc.)"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    g_state["lab_clients"].add(ws)
    print(f"🔌 Cliente de LABORATORIO conectado: {request.remote}")

    # Enviar estado inicial al conectar
    await ws.send(json.dumps({
        "type": "init",
        "checkpoints": get_checkpoints(),
        "is_training": g_state["training_process"] is not None,
        "is_simulating": g_state["simulation_process"] is not None
    }))

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                command = data.get("command")
                
                if command == "start_training":
                    if g_state["training_process"]:
                        await ws.send(json.dumps({"type": "error", "data": "El entrenamiento ya está en curso."}))
                        continue
                        
                    print("🚀 Recibida orden de ENTRENAR")
                    # Construir el comando
                    args = data.get("args", {})
                    cmd = [
                        sys.executable, # El ejecutable de python actual
                        "train.py",
                        "--name", args.get("name", cfg.EXPERIMENT_NAME),
                        "--model", args.get("model", "unet"),
                        "--lr", str(args.get("lr", cfg.LR_RATE_M)),
                        "--episodes", str(args.get("episodes", cfg.EPISODES_TO_ADD)),
                        "--hidden_channels", str(args.get("hidden_channels", cfg.HIDDEN_CHANNELS))
                    ]
                    
                    # Lanzar el subproceso
                    g_state["training_process"] = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT
                    )
                    
                    await broadcast_lab_message({"type": "status", "data": "Iniciando entrenamiento..."})
                    
                    # Iniciar la tarea de retransmisión de logs
                    asyncio.create_task(stream_subprocess_logs(g_state["training_process"], ws))

                elif command == "stop_training":
                    if g_state["training_process"]:
                        print("🛑 Recibida orden de DETENER ENTRENAMIENTO")
                        g_state["training_process"].terminate()
                        await g_state["training_process"].wait()
                        g_state["training_process"] = None
                        await broadcast_lab_message({"type": "status", "data": "Entrenamiento detenido por el usuario."})
                    else:
                        await ws.send(json.dumps({"type": "error", "data": "No hay ningún entrenamiento en curso."}))

                elif command == "start_simulation":
                    # (Esta es la versión simple: simplemente lanza el otro servidor)
                    if g_state["simulation_process"]:
                        await ws.send(json.dumps({"type": "error", "data": "La simulación ya está en curso."}))
                        continue
                    
                    print("🛰️ Recibida orden de SIMULAR")
                    # (Aquí podríamos lanzar 'main.py' como un subproceso,
                    # pero es más limpio que este servidor también maneje la simulación)
                    # Por simplicidad, esta UI solo manejará el ENTRENAMIENTO por ahora.
                    # El servidor de simulación lo sigues corriendo con 'python3 main.py'
                    await ws.send(json.dumps({"type": "error", "data": "Función no implementada. Corre 'python3 main.py' por separado."}))

                elif command == "refresh_checkpoints":
                    await ws.send(json.dumps({
                        "type": "checkpoints",
                        "data": get_checkpoints()
                    }))

            elif msg.type == web.WSMsgType.ERROR:
                print(f"Error de WebSocket (Lab): {ws.exception()}")
                
    finally:
        print(f"🔌 Cliente de LABORATORIO desconectado: {request.remote}")
        g_state["lab_clients"].remove(ws)
    return ws


# --- Punto de Entrada del Servidor de Laboratorio ---
async def main_lab_server():
    app = web.Application()
    
    # Añadir rutas
    app.router.add_get('/', serve_lab_html) # Sirve la UI del Laboratorio
    app.router.add_get('/visor', serve_viewer_html) # Sirve el visor de simulación
    app.router.add_get('/ws_lab', lab_websocket_handler) # WebSocket del Laboratorio
    
    print(f"--- ✅ Servidor de LABORATORIO iniciado en http://{cfg.LAB_SERVER_HOST}:{cfg.LAB_SERVER_PORT} ---")
    print("Abre esa URL en tu navegador para ver el panel de control.")
    
    # Correr el servidor
    web.run_app(app, host=cfg.LAB_SERVER_HOST, port=cfg.LAB_SERVER_PORT)

if __name__ == "__main__":
    # Importante: Borra tu 'main.py' y renombra este archivo a 'main.py'
    # O simplemente ejecuta: python3 lab_server.py
    try:
        asyncio.run(main_lab_server())
    except KeyboardInterrupt:
        print("\n🛑 Servidor de Laboratorio detenido.")