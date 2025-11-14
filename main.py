import os
import logging
from aiohttp import web
# Importamos el objeto 'routes' que ahora contiene TODAS nuestras rutas (API y WebSocket)
from src.server_handlers import routes
from src.server_state import g_state # Estado global si es necesario inicializar algo

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def serve_viewer(request):
    """Sirve el archivo principal de la interfaz de usuario."""
    viewer_path = os.path.join(os.path.dirname(__file__), 'viewer.html')
    try:
        return web.FileResponse(viewer_path)
    except FileNotFoundError:
        return web.Response(text="viewer.html no encontrado.", status=404)

async def on_startup(app):
    """Inicializa los recursos de la aplicación al arrancar."""
    # Inicializamos el conjunto de clientes WebSocket en el estado global
    app['g_state'] = g_state
    app['g_state']['clients'] = set()
    logging.info("Servidor Aetheria iniciado. Estado inicializado.")

async def on_shutdown(app):
    """Limpia los recursos de la aplicación al apagar."""
    # Cierra todas las conexiones WebSocket activas
    for ws in app['g_state']['clients']:
        if not ws.closed:
            await ws.close(code=1001, message='Server shutdown')
    logging.info("Servidor Aetheria apagándose. Conexiones WebSocket cerradas.")

def main():
    """Función principal para configurar y lanzar el servidor web."""
    app = web.Application()

    # Añadimos la ruta para servir el viewer.html en la raíz
    app.router.add_get("/", serve_viewer)

    # Registramos TODAS las rutas importadas desde server_handlers.py
    # Esto incluye /api/experiments y /ws
    app.add_routes(routes)

    # Registramos las funciones de startup y shutdown
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    # Obtenemos el puerto de las variables de entorno o usamos un valor por defecto
    port = int(os.environ.get("AETHERIA_PORT", 8080))

    logging.info(f"El visor de Aetheria estará disponible en http://localhost:{port}")
    web.run_app(app, host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()
