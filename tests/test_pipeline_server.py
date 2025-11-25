# scripts/test_pipeline_server.py
import sys
import os
import logging
from aiohttp import web

# Configurar el logging para ver los mensajes de depuración
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Asegurarse de que el directorio raíz del proyecto esté en el path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.pipeline_server import setup_app
    logging.info("Intentando configurar la aplicación del servidor...")
    app = setup_app()
    if isinstance(app, web.Application):
        logging.info("✅ La aplicación del servidor se configuró correctamente.")
        logging.info(f"Rutas registradas: {len(app.router._routes)}")
        # Puedes añadir más aserciones aquí si quieres probar rutas específicas
    else:
        logging.error("❌ setup_app no devolvió una instancia de web.Application.")

except ImportError as e:
    logging.error(f"❌ ImportError al importar pipeline_server: {e}")
    logging.error("Asegúrate de que el PYTHONPATH esté configurado correctamente o que no haya dependencias circulares.")
except Exception as e:
    logging.error(f"❌ Error inesperado durante la configuración del servidor: {e}", exc_info=True)

logging.info("Test de pipeline_server.py finalizado.")
