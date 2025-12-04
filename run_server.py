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
    try:
        from src.pipelines import pipeline_server
        log.info("Módulo 'src.pipelines.pipeline_server' importado exitosamente.")
    except Exception as e:
        log.error(f"Error importando pipeline_server: {e}", exc_info=True)
        raise e

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
        from src.server.server_state import g_state
        from src.server.server_handlers import create_experiment_handler
        
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
    import argparse
    
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Servidor Aetheria")
    parser.add_argument('--no-frontend', action='store_true', 
                       help='No servir el frontend estático, solo WebSocket API')
    parser.add_argument('--port', type=int, default=None,
                       help='Puerto del servidor (por defecto: desde config)')
    parser.add_argument('--host', type=str, default=None,
                       help='Host del servidor (por defecto: desde config)')
    parser.add_argument('--autostart', action='store_true',
                       help='Inicia automáticamente una simulación dummy para pruebas')
    args = parser.parse_args()
    
    # Configurar variable de entorno según --no-frontend
    # CRÍTICO: Si NO se pasa --no-frontend, limpiar la variable para asegurar que el frontend se sirva
    if args.no_frontend:
        os.environ['ATHERIA_NO_FRONTEND'] = '1'
        log.info("Frontend desactivado (--no-frontend). Solo WebSocket API disponible.")
    else:
        # Si NO se pasa --no-frontend, asegurar que la variable NO esté establecida
        # Esto es importante porque puede estar establecida desde una ejecución anterior
        if 'ATHERIA_NO_FRONTEND' in os.environ:
            del os.environ['ATHERIA_NO_FRONTEND']
            log.info("Frontend activado (--no-frontend no especificado).")
    
    log.info("Pasando el control a 'pipeline_server.main()'.")
    
    # Variables globales para almacenar el event loop y el shutdown event
    shutdown_event = None
    main_loop = None
    shutdown_in_progress = False
    
    # --- Auto-start Simulation Task (Temporary/Dev) ---
    async def autostart_simulation_task():
        """Initializes a dummy motor and starts simulation if requested."""
        if not getattr(args, 'autostart', False):
            return

        try:
            log.info("⚡ Autostart: Esperando a que el sistema se estabilice...")
            await asyncio.sleep(5)

            from src.server.server_state import g_state
            from src.config import DEVICE
            from src.motor_factory import get_motor
            import torch.nn as nn

            if g_state.get('motor'):
                log.info("⚡ Autostart: Motor ya existe, solo reanudando.")
                g_state['is_paused'] = False
                return

            log.info("⚡ Autostart: Inicializando motor dummy para pruebas...")

            # Create a minimal dummy model
            class DummyModel(nn.Module):
                def forward(self, x):
                    return x # Identity

            dummy_model = DummyModel().to(DEVICE)
            dummy_config = {
                'ENGINE_TYPE': 'CARTESIAN',
                'GRID_SIZE': 64,
                'D_STATE': 8
            }

            motor = get_motor(dummy_config, DEVICE, dummy_model)

            # Initialize state using state object, not motor
            # CartesianEngine has a 'state' attribute which is QuantumState
            if hasattr(motor, 'state') and hasattr(motor.state, '_initialize_state'):
                 # Re-initialize the psi within the state
                 motor.state.psi = motor.state._initialize_state(mode='random')

            # Update global state
            g_state['motor'] = motor
            g_state['is_paused'] = False
            g_state['viz_type'] = 'density'
            g_state['simulation_step'] = 0

            log.info("⚡ Autostart: Simulación iniciada automáticamente.")

        except Exception as e:
            log.error(f"❌ Error en autostart: {e}", exc_info=True)

    # Configurar handler de señales para guardar estado antes de cerrar
    def signal_handler(signum, frame):
        global shutdown_in_progress
        if shutdown_in_progress:
            log.warning(f"Señal {signum} recibida durante shutdown. Forzando salida inmediata...")
            # Si ya estamos en proceso de shutdown y recibimos otra señal (Ctrl+C nuevamente), salir inmediatamente
            import os
            os._exit(1)  # Salida forzada, no llama finally
        
        log.info(f"Señal {signum} recibida. Iniciando shutdown graceful (máx. 5 segundos)...")
        shutdown_in_progress = True
        
        # Configurar el evento en el event loop actual
        if shutdown_event and main_loop:
            main_loop.call_soon_threadsafe(shutdown_event.set)
        else:
            log.warning("Shutdown event o main loop no están disponibles. Forzando salida...")
            import os
            os._exit(1)
        
        # Si después de 5 segundos no se ha cerrado, forzar salida
        import threading
        def force_exit_after_timeout():
            import time
            time.sleep(5)
            if shutdown_in_progress:
                log.warning("Shutdown excedió 5 segundos. Forzando salida...")
                import os
                os._exit(1)
        
        timeout_thread = threading.Thread(target=force_exit_after_timeout, daemon=True)
        timeout_thread.start()
    
    # Registrar handlers de señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_with_shutdown():
        global shutdown_event, main_loop, shutdown_in_progress
        shutdown_event = asyncio.Event()
        main_loop = asyncio.get_running_loop()
        
        try:
            # Pasar configuración de frontend al servidor
            serve_frontend = not args.no_frontend

            # Crear tarea de autostart si corresponde
            if args.autostart:
                asyncio.create_task(autostart_simulation_task())

            await pipeline_server.main(shutdown_event, serve_frontend=serve_frontend)
        except asyncio.CancelledError:
            log.info("Tareas canceladas durante shutdown.")
        except Exception as e:
            log.error(f"Error durante ejecución del servidor: {e}", exc_info=True)
        finally:
            shutdown_in_progress = False
            log.info("Servidor cerrado completamente.")
    
    try:
        # Usar el main del módulo importado pasando el evento de shutdown
        # El main debe manejar el shutdown graceful
        asyncio.run(run_with_shutdown())
    except KeyboardInterrupt:
        log.info("Cierre solicitado por el usuario (Ctrl+C).")
    except Exception as e:
        log.error(f"Una excepción no controlada ha detenido el servidor: {e}", exc_info=True)
