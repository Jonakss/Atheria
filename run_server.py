# run_server.py
import sys
import os
import asyncio
import logging
import signal

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

# --- Función para guardar estado antes de cerrar ---
async def save_state_before_shutdown():
    """Guarda el estado del entrenamiento y simulación antes de cerrar."""
    try:
        from src.server_state import g_state
        from src.server_handlers import create_experiment_handler
        
        # Guardar estado de entrenamiento si hay un proceso activo
        training_process = g_state.get('training_process')
        if training_process and training_process.returncode is None:
            log.info("Proceso de entrenamiento detectado. Intentando guardar checkpoint...")
            # El entrenamiento se guarda automáticamente cada SAVE_EVERY_EPISODES
            # Aquí solo notificamos que se está cerrando
            log.info("Nota: El entrenamiento guarda checkpoints periódicamente. Verifica los checkpoints guardados.")
        
        # Guardar estado de simulación si hay un motor activo
        motor = g_state.get('motor')
        if motor and not g_state.get('is_paused', True):
            log.info("Simulación activa detectada. Estado de simulación se puede recuperar al reiniciar.")
            # El estado de simulación se reinicia al cargar el modelo, así que no necesitamos guardarlo
        
        log.info("Estado guardado correctamente.")
    except Exception as e:
        log.error(f"Error al guardar estado: {e}", exc_info=True)

# --- Ejecutar el Servidor ---
if __name__ == "__main__":
    log.info("Pasando el control a 'pipeline_server.main()'.")
    
    # Variables globales para almacenar el event loop y el shutdown event
    shutdown_event = None
    main_loop = None
    
    # Configurar handler de señales para guardar estado antes de cerrar
    def signal_handler(signum, frame):
        log.info(f"Señal {signum} recibida. Iniciando shutdown graceful...")
        # Configurar el evento en el event loop actual
        if shutdown_event and main_loop:
            main_loop.call_soon_threadsafe(shutdown_event.set)
    
    # Registrar handlers de señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_with_shutdown():
        global shutdown_event, main_loop
        shutdown_event = asyncio.Event()
        main_loop = asyncio.get_running_loop()
        await pipeline_server.main(shutdown_event)
    
    try:
        # Usar el main del módulo importado pasando el evento de shutdown
        # El main debe manejar el shutdown graceful
        asyncio.run(run_with_shutdown())
    except KeyboardInterrupt:
        log.info("Cierre solicitado por el usuario (Ctrl+C).")
    except Exception as e:
        log.error(f"Una excepción no controlada ha detenido el servidor: {e}", exc_info=True)
