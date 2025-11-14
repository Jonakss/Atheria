# run_server.py
import sys
import os
import asyncio
import logging

# --- Configuración de Logging Temprana ---
# Esto es para capturar errores incluso antes de que el servidor principal se cargue.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("BOOTSTRAP")

log.info("Iniciando el script de arranque 'run_server.py'...")

# --- ¡¡CLAVE!! Añadir el directorio del proyecto al path de Python ---
# Esto asegura que los imports como 'from src.pipeline_server' funcionen sin ambigüedad.
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    log.info(f"Añadiendo el directorio del proyecto al path de Python: {project_root}")
    sys.path.insert(0, project_root)

    # Ahora que el path está configurado, podemos importar nuestros módulos de forma segura.
    from src import pipeline_server
    log.info("Módulo 'src.pipeline_server' importado exitosamente.")

except ImportError as e:
    log.error("="*60)
    log.error("¡¡ERROR CRÍTICO DE IMPORTACIÓN!!")
    log.error(f"No se pudo importar 'src.pipeline_server'. Error: {e}")
    log.error("Asegúrate de que la estructura de directorios es correcta y que 'src/__init__.py' existe.")
    log.error("="*60)
    sys.exit(1)

# --- Ejecutar el Servidor ---
if __name__ == "__main__":
    log.info("Pasando el control a 'pipeline_server.main()'.")
    try:
        # Usar el main del módulo importado
        asyncio.run(pipeline_server.main())
    except KeyboardInterrupt:
        log.info("Cierre solicitado por el usuario (Ctrl+C) desde run_server.py.")
    except Exception as e:
        log.error(f"Una excepción no controlada ha detenido el servidor: {e}", exc_info=True)
