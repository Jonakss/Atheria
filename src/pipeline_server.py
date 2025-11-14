# src/pipeline_server.py
import asyncio
import logging
import os
import signal
from aiohttp import web
from . import config as cfg
from .server_handlers import websocket_handler, simulation_loop
from .server_state import g_state # ¡Importar g_state!

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

async def handle_index(request):
    index_path = os.path.join(cfg.FRONTEND_DIST_PATH, 'index.html')
    return web.FileResponse(index_path)

async def main():
    app = web.Application()
    app.router.add_get("/ws", websocket_handler)

    is_development = os.environ.get("AETHERIA_ENV") == "development"

    if is_development:
        log.info("Modo DESARROLLO: El servidor backend se está ejecutando.")
    else:
        if not os.path.exists(cfg.FRONTEND_DIST_PATH):
            log.error(f"Modo PRODUCCIÓN: Directorio '{cfg.FRONTEND_DIST_PATH}' no encontrado.")
            return
        log.info(f"Modo PRODUCCIÓN: Sirviendo frontend desde {cfg.FRONTEND_DIST_PATH}")
        app.router.add_get('/', handle_index)
        app.router.add_static('/', path=cfg.FRONTEND_DIST_PATH, name='static')

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, cfg.LAB_SERVER_HOST, cfg.LAB_SERVER_PORT)
    await site.start()
    log.info(f"🚀 Servidor AETHERIA ejecutándose en http://{cfg.LAB_SERVER_HOST}:{cfg.LAB_SERVER_PORT}")

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    simulation_task = loop.create_task(simulation_loop())
    
    def shutdown(sig):
        log.info(f"Señal de parada recibida ({sig.name}), iniciando cierre limpio...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    log.info("Presiona CTRL+C para detener.")
    
    try:
        await stop_event.wait()
    finally:
        log.info("Cerrando tareas de fondo y servidor...")
        
        # --- ¡¡CORRECCIÓN CLAVE!! Cierre Asertivo de Sockets ---
        # Cerrar explícitamente todas las conexiones de clientes WebSocket.
        # Esto rompe el deadlock y permite un cierre rápido.
        log.info(f"Cerrando {len(g_state['websockets'])} conexiones de clientes...")
        close_tasks = [ws.close() for ws in g_state['websockets']]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        
        simulation_task.cancel()
        await runner.cleanup()
        
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Servidor detenido limpiamente.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Cierre forzado por KeyboardInterrupt.")
