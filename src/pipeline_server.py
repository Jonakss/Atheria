# src/pipeline_server.py
import asyncio
import logging
import os
import sys
from aiohttp import web

# --- Configuración del Path ---
# Asegurarse de que el directorio raíz del proyecto esté en el path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Importaciones del Proyecto (Ahora modularizadas) ---
from src import config as cfg
from src.server_state import g_state
from src.server_handlers import websocket_handler, start_simulation, stop_simulation

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuración y Punto de Entrada del Servidor ---
FRONTEND_DEV_SERVER_URL = "http://localhost:5173"
FRONTEND_BUILD_DIR = os.path.join(PROJECT_ROOT, "frontend", "dist")

async def frontend_dev_proxy(request):
    """Redirige las peticiones al servidor de desarrollo de Vite."""
    path = request.match_info.get('path', '')
    url = f"{FRONTEND_DEV_SERVER_URL}/{path}"
    if request.query_string:
        url += f"?{request.query_string}"

    try:
        async with web.ClientSession() as session:
            async with session.get(url) as resp:
                raw_content = await resp.read()
                return web.Response(body=raw_content, status=resp.status, headers=resp.headers)
    except Exception as e:
        logging.error(f"Error en el proxy a Vite: {e}", exc_info=True)
        return web.Response(text=f"Error: No se pudo conectar al servidor de desarrollo de Vite en {FRONTEND_DEV_SERVER_URL}. ¿Está en ejecución?", status=503)

async def cleanup_background_tasks(app):
    """Cancela todas las tareas de simulación al apagar el servidor."""
    logging.info("Limpiando tareas de fondo...")
    stop_simulation() # Usa la función de control para limpiar todo.

def setup_app():
    """Configura y devuelve la aplicación aiohttp."""
    app = web.Application()
    app.on_cleanup.append(cleanup_background_tasks)
    
    # Rutas principales
    app.router.add_get('/ws', websocket_handler)

    # Servir el frontend en modo desarrollo o producción
    env = os.environ.get("AETHERIA_ENV", "production")
    if env == "development":
        logging.info(f"Modo DESARROLLO: Redirigiendo frontend a {FRONTEND_DEV_SERVER_URL}")
        app.router.add_get('/{path:.*}', frontend_dev_proxy)
    else:
        logging.info(f"Modo PRODUCCIÓN: Sirviendo frontend desde {FRONTEND_BUILD_DIR}")
        if not os.path.isdir(FRONTEND_BUILD_DIR) or not os.path.exists(os.path.join(FRONTEND_BUILD_DIR, 'index.html')):
            logging.error(f"Directorio de build del frontend no encontrado o 'index.html' ausente en: {FRONTEND_BUILD_DIR}")
            logging.error("Por favor, ejecuta 'npm run build' en el directorio 'frontend' antes de iniciar en modo producción.")
            # Podríamos tener una página de error simple aquí
            return app # Devuelve la app sin rutas de frontend para evitar que se cuelgue

        # 1. Sirve los assets (JS, CSS) desde su carpeta específica
        app.router.add_static('/assets', path=os.path.join(FRONTEND_BUILD_DIR, 'assets'), name='assets')
        
        # 2. Para cualquier otra ruta, sirve el index.html principal.
        #    Esto permite que el enrutador de React (client-side) tome el control.
        async def serve_index(request):
            return web.FileResponse(os.path.join(FRONTEND_BUILD_DIR, 'index.html'))
            
        app.router.add_get('/{path:.*}', serve_index)

    return app

if __name__ == "__main__":
    logging.info("Iniciando servidor de Aetheria...")
    app = setup_app()
    web.run_app(app, host=cfg.LAB_SERVER_HOST, port=cfg.LAB_SERVER_PORT)
    logging.info("Servidor de Aetheria detenido.")