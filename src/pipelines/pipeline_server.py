# src/pipeline_server.py
import asyncio
import json
import logging
import os
import uuid
from aiohttp import web
from pathlib import Path
import torch

# Asumimos la existencia y correcto funcionamiento de estos módulos locales
from .. import config as global_cfg
from ..server.server_state import g_state, broadcast, send_notification, send_to_websocket, optimize_frame_payload, get_payload_size, apply_roi_to_payload
from ..utils import get_experiment_list, load_experiment_config, get_latest_checkpoint
from .viz import get_visualization_data
from .core.simulation_loop import simulation_loop
from .core.websocket_handler import websocket_handler as ws_handler
from .core.helpers import calculate_adaptive_downsample, calculate_adaptive_roi
from .core.status_helpers import build_inference_status_payload

# Importar handlers desde el paquete handlers
from .handlers import (
    EXPERIMENT_HANDLERS, 
    SIMULATION_HANDLERS, 
    INFERENCE_HANDLERS, 
    SYSTEM_HANDLERS,
    ANALYSIS_HANDLERS,
    HISTORY_HANDLERS
)

# Configuración de logging - Reducir verbosidad en producción
# INFO para eventos importantes, DEBUG para detalles del bucle de simulación
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# --- WEBSOCKET HANDLER ---
# Ahora importado desde core.websocket_handler
def create_websocket_handler(handlers):
    """Factory function para crear websocket_handler con handlers."""
    async def handler(request):
        return await ws_handler(request, handlers)
    return handler

# Mantener función original por compatibilidad (ahora es wrapper)
async def websocket_handler(request):
    """Maneja las conexiones WebSocket entrantes."""
    # Logging para debugging en entornos con proxy (solo debug para reducir verbosidad)
    client_ip = request.headers.get('X-Forwarded-For', request.remote)
    logging.debug(f"Intento de conexión WebSocket desde {client_ip}")
    
    ws = web.WebSocketResponse()
    
    # Manejar errores durante la preparación de la conexión (reconexiones rápidas, etc.)
    try:
        await ws.prepare(request)
    except (ConnectionResetError, ConnectionError, OSError) as e:
        # El cliente se desconectó antes de establecer la conexión - esto es normal
        # No loguear como error, solo como debug
        logging.debug(f"Conexión WebSocket cancelada durante preparación: {type(e).__name__}")
        # Retornar respuesta HTTP normal en lugar de WebSocket no preparado
        return web.Response(status=101, text="WebSocket connection cancelled")
    except Exception as e:
        # Otros errores inesperados
        logging.error(f"Error preparando conexión WebSocket: {e}", exc_info=True)
        logging.error(f"Headers de la solicitud: {dict(request.headers)}")
        # Retornar respuesta HTTP de error en lugar de WebSocket no preparado
        return web.Response(status=500, text=f"WebSocket connection failed: {str(e)}")
    
    ws_id = str(uuid.uuid4())
    g_state['websockets'][ws_id] = ws
    logging.debug(f"Nueva conexión WebSocket: {ws_id}")
    
    # Enviar estado inicial al cliente
    experiments = get_experiment_list()
    
    # Obtener versiones de los engines incluso sin modelo cargado
    compile_status_without_model = None
    try:
        # Intentar obtener versiones de los engines disponibles
        from ..engines.qca_engine import Aetheria_Motor
        python_version = getattr(Aetheria_Motor, 'VERSION', None) or getattr(Aetheria_Motor, 'get_version', lambda: "unknown")() if hasattr(Aetheria_Motor, 'get_version') else "unknown"
        
        native_version = None
        wrapper_version = None
        try:
            import atheria_core
            # Obtener versión del motor nativo si está disponible
            from ..engines.native_engine_wrapper import NativeEngineWrapper
            wrapper_version = getattr(NativeEngineWrapper, 'VERSION', None) or "unknown"
            # Intentar obtener versión del motor C++ directamente del módulo
            try:
                if hasattr(atheria_core, 'get_version'):
                    native_version = atheria_core.get_version()
                elif hasattr(atheria_core, '__version__'):
                    native_version = getattr(atheria_core, '__version__')
                else:
                    native_version = "available"  # Solo indicar que está disponible
            except Exception as e:
                logging.debug(f"No se pudo obtener versión del motor nativo en estado inicial: {e}")
                native_version = "available"  # Solo indicar que está disponible
        except (ImportError, OSError, RuntimeError):
            pass
        
        compile_status_without_model = {
            "is_compiled": False,
            "is_native": False,
            "model_name": "None",
            "compiles_enabled": True,
            "device_str": None,
            "native_version": native_version,
            "wrapper_version": wrapper_version,
            "python_version": python_version
        }
    except Exception as e:
        logging.debug(f"No se pudieron obtener versiones de engines: {e}")
    
    initial_state = {
        "type": "initial_state",
        "payload": {
            "experiments": experiments,
            "training_status": "running" if g_state.get('training_process') else "idle",
            "inference_status": "running" if not g_state.get('is_paused', True) else "paused"
        }
    }
    
    # Si hay compile_status sin modelo, incluirlo en el estado inicial
    if compile_status_without_model:
        initial_state["payload"]["compile_status"] = compile_status_without_model
    # Enviar estado inicial - manejar errores de conexión
    try:
        await ws.send_json(initial_state)
    except (ConnectionResetError, ConnectionError, OSError) as e:
        # Cliente ya se desconectó - limpiar y retornar
        if ws_id in g_state['websockets']:
            del g_state['websockets'][ws_id]
        logging.debug(f"Conexión WebSocket cerrada antes de enviar estado inicial: {type(e).__name__}")
        return ws
    except Exception as e:
        logging.warning(f"Error enviando estado inicial a WebSocket {ws_id}: {e}")
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    scope = data.get('scope')
                    command = data.get('command')
                    args = data.get('args', {})
                    args['ws_id'] = ws_id
                    
                    logging.debug(f"Comando recibido: {scope}.{command} de [{ws_id}]")
                    
                    # Buscar y ejecutar el handler correspondiente
                    if scope in HANDLERS and command in HANDLERS[scope]:
                        handler = HANDLERS[scope][command]
                        await handler(args)
                    else:
                        logging.warning(f"Comando desconocido: {scope}.{command}")
                        await send_notification(ws, f"Comando desconocido: {scope}.{command}", "error")
                        
                except json.JSONDecodeError:
                    logging.error(f"Error al parsear JSON del mensaje: {msg.data}")
                    await send_notification(ws, "Error al procesar el comando.", "error")
                except Exception as e:
                    logging.error(f"Error procesando comando: {e}", exc_info=True)
                    try:
                        await send_notification(ws, f"Error al ejecutar comando: {str(e)}", "error")
                    except (ConnectionResetError, ConnectionError, OSError):
                        # Cliente ya se desconectó
                        break
            elif msg.type == web.WSMsgType.ERROR:
                exception = ws.exception()
                if exception:
                    # Solo loguear errores reales, no desconexiones normales
                    if isinstance(exception, (ConnectionResetError, ConnectionError, OSError)):
                        logging.debug(f"WebSocket {ws_id} desconectado: {type(exception).__name__}")
                    else:
                        logging.error(f"Error en WebSocket {ws_id}: {exception}")
            elif msg.type == web.WSMsgType.CLOSE:
                # Desconexión limpia
                logging.debug(f"WebSocket {ws_id} cerrado normalmente")
    except (ConnectionResetError, ConnectionError, OSError) as e:
        # Desconexión durante el loop - normal durante reconexiones
        logging.debug(f"Conexión WebSocket {ws_id} interrumpida: {type(e).__name__}")
    except Exception as e:
        logging.error(f"Error inesperado en WebSocket {ws_id}: {e}", exc_info=True)
    finally:
        # Limpiar la conexión - solo si el WebSocket fue preparado correctamente
        if ws_id in g_state['websockets']:
            del g_state['websockets'][ws_id]
        # Verificar que el WebSocket esté preparado antes de intentar cerrarlo
        if not ws.closed:
            try:
                await ws.close()
            except (RuntimeError, ConnectionResetError, ConnectionError, OSError) as e:
                # WebSocket ya estaba cerrado o no fue preparado - esto es normal
                logging.debug(f"WebSocket {ws_id} ya estaba cerrado o no preparado: {type(e).__name__}")
        logging.debug(f"Conexión WebSocket cerrada: {ws_id}")
    
    return ws


# Diccionario central para mapear comandos a funciones handler
# Combinar handlers de módulos extraídos
HANDLERS = {
    "experiment": EXPERIMENT_HANDLERS,
    "simulation": SIMULATION_HANDLERS,
    "analysis": ANALYSIS_HANDLERS,
    "inference": INFERENCE_HANDLERS,
    "server": SYSTEM_HANDLERS,
    "system": SYSTEM_HANDLERS,
    "history": HISTORY_HANDLERS
}

# --- Configuración de la App aiohttp ---

def setup_routes(app, serve_frontend=True):
    """
    Configura las rutas del servidor.
    
    Args:
        app: Aplicación web de aiohttp
        serve_frontend: Si True, sirve el frontend estático. Si False, solo WebSocket.
                       Por defecto True. Se puede desactivar con --no-frontend o variable de entorno ATHERIA_NO_FRONTEND=1
    """
    # Usar FRONTEND_DIST_PATH desde config para asegurar consistencia
    STATIC_FILES_ROOT = Path(global_cfg.FRONTEND_DIST_PATH) if hasattr(global_cfg, 'FRONTEND_DIST_PATH') else Path(__file__).parent.parent.parent.resolve() / 'frontend' / 'dist'
    
    # Siempre agregar la ruta WebSocket (debe tener prioridad absoluta)
    # Esto permite que el servidor funcione aunque no tenga el frontend construido
    # Crear handler con HANDLERS (ya está definido antes de llamar a setup_routes)
    ws_handler_func = create_websocket_handler(HANDLERS)
    app.router.add_get("/ws", ws_handler_func)
    
    # Si se desactiva el frontend o no existe, servir solo mensaje informativo
    if not serve_frontend:
        logging.info("Frontend desactivado. Servidor funcionará solo con WebSocket (--no-frontend).")
        async def serve_info(request):
            return web.Response(
                text="Aetheria Server está funcionando. WebSocket disponible en /ws\n\n"
                     "Frontend desactivado. Solo API WebSocket disponible.",
                content_type='text/plain'
            )
        app.router.add_get('/', serve_info)
        return
    
    # Verificar si el frontend existe
    if not STATIC_FILES_ROOT.exists() or not (STATIC_FILES_ROOT / 'index.html').exists():
        logging.warning(f"Directorio de frontend '{STATIC_FILES_ROOT}' no encontrado o incompleto.")
        logging.warning("El servidor funcionará solo con WebSocket. Para servir el frontend, ejecuta 'npm run build' en la carpeta 'frontend'.")
        logging.warning("O usa --no-frontend para desactivar explícitamente el frontend.")
        
        # Servir una respuesta simple en la raíz para indicar que el servidor está funcionando
        async def serve_info(request):
            return web.Response(
                text="Aetheria Server está funcionando. WebSocket disponible en /ws\n\n"
                     "Para servir el frontend, construye los archivos estáticos con 'npm run build' en la carpeta 'frontend'.",
                content_type='text/plain'
            )
        app.router.add_get('/', serve_info)
        return

    logging.info(f"Sirviendo archivos estáticos desde: {STATIC_FILES_ROOT}")
    
    # Servir index.html en la raíz
    async def serve_index(request):
        return web.FileResponse(STATIC_FILES_ROOT / 'index.html')
    app.router.add_get('/', serve_index)
    
    # Ruta "catch-all" que maneja tanto archivos estáticos como rutas del SPA
    # Esencial para Single Page Applications (SPA) como React que usan routing del lado del cliente.
    async def serve_static_or_spa(request):
        path = request.match_info.get('path', '')
        # Limpiar el path (remover barras iniciales y prevenir path traversal)
        path = path.lstrip('/')
        if not path:
            return web.FileResponse(STATIC_FILES_ROOT / 'index.html')
        
        file_path = STATIC_FILES_ROOT / path
        
        # Verificar que el archivo esté dentro del directorio estático (seguridad)
        try:
            file_path.resolve().relative_to(STATIC_FILES_ROOT.resolve())
        except ValueError:
            # Path traversal attempt, servir index.html
            return web.FileResponse(STATIC_FILES_ROOT / 'index.html')
        
        # Si es un archivo que existe, servirlo
        if file_path.is_file() and file_path.exists():
            return web.FileResponse(file_path)
        
        # Si no existe, servir index.html (para rutas del SPA como /experiments, /training, etc.)
        return web.FileResponse(STATIC_FILES_ROOT / 'index.html')
    
    app.router.add_get('/{path:.*}', serve_static_or_spa)


async def on_startup(app):
    """Crea la tarea del bucle de simulación cuando el servidor arranca."""
    app['simulation_loop'] = asyncio.create_task(simulation_loop())

async def on_shutdown(app):
    """Cancela la tarea del bucle de simulación cuando el servidor se apaga y guarda el estado."""
    logging.info("Iniciando cierre ordenado del servidor...")
    
    # PRIMERO: Cerrar todas las conexiones WebSocket activas de forma agresiva
    websockets = list(g_state.get('websockets', {}).items())  # Lista de tuplas para evitar problemas de closure
    if websockets:
        logging.info(f"Cerrando {len(websockets)} conexiones WebSocket activas...")
        
        # Función helper para cerrar un WebSocket con timeout
        async def close_ws_safe(ws_id, ws):
            try:
                if ws.closed:
                    return
                # Intentar cerrar de forma ordenada con timeout corto
                await asyncio.wait_for(ws.close(code=1001, message=b'Server shutting down'), timeout=0.5)
            except asyncio.TimeoutError:
                # Timeout: intentar cerrar el transporte directamente
                try:
                    if hasattr(ws, '_writer') and ws._writer:
                        ws._writer.close()
                        await ws._writer.wait_closed()
                except Exception:
                    pass
            except Exception as e:
                logging.debug(f"Error cerrando WebSocket {ws_id}: {e}")
        
        # Ejecutar todos los cierres en paralelo con timeout total muy corto (1 segundo)
        close_tasks = [close_ws_safe(ws_id, ws) for ws_id, ws in websockets if not ws.closed]
        if close_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*close_tasks, return_exceptions=True), timeout=1.0)
            except asyncio.TimeoutError:
                logging.warning("Timeout cerrando WebSockets, continuando con shutdown...")
        
        # Limpiar el diccionario de WebSockets (siempre, incluso si falló el cierre)
        g_state['websockets'].clear()
        logging.info("Conexiones WebSocket cerradas/limpiadas")
    
    # Cancelar el bucle de simulación PRIMERO (para evitar que siga generando frames)
    if 'simulation_loop' in app:
        app['simulation_loop'].cancel()
        try:
            await asyncio.wait_for(app['simulation_loop'], timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logging.info("Bucle de simulación detenido")
    
    # Detener proceso de entrenamiento si está activo
    training_process = g_state.get('training_process')
    if training_process and training_process.returncode is None:
        logging.info("Deteniendo proceso de entrenamiento...")
        try:
            # Enviar señal SIGTERM para permitir que el proceso guarde su checkpoint
            training_process.terminate()
            # Esperar un poco para que el proceso pueda guardar
            try:
                await asyncio.wait_for(asyncio.to_thread(training_process.wait), timeout=3.0)
                logging.info("Proceso de entrenamiento detenido correctamente.")
            except asyncio.TimeoutError:
                logging.warning("El proceso de entrenamiento no respondió en 3 segundos. Forzando cierre...")
                training_process.kill()
                await asyncio.to_thread(training_process.wait)
        except ProcessLookupError:
            logging.warning("El proceso de entrenamiento ya había terminado.")
        except Exception as e:
            logging.error(f"Error al detener proceso de entrenamiento: {e}")
        finally:
            g_state['training_process'] = None
    
    # Guardar estado de simulación si hay un motor activo
    motor = g_state.get('motor')
    if motor and not g_state.get('is_paused', True):
        logging.info("Pausando simulación antes de cerrar...")
        try:
            # Pausar la simulación
            g_state['is_paused'] = True
            logging.info("Simulación pausada")
        except Exception as e:
            logging.error(f"Error al pausar simulación: {e}")
    
    logging.info("Cierre ordenado completado.")

async def main(shutdown_event=None, serve_frontend=None):
    """
    Función principal para configurar e iniciar el servidor web.
    
    Args:
        shutdown_event: Evento de asyncio para señalizar shutdown (opcional)
        serve_frontend: Si True, sirve el frontend estático. Si None, auto-detecta desde variable de entorno.
                       Por defecto True si no se especifica.
    """
    # Exponer shutdown_event en g_state para que los handlers puedan acceder
    if shutdown_event:
        g_state['shutdown_event'] = shutdown_event
    
    app = web.Application()
    
    # Determinar si servir frontend
    # 1. Si se pasa explícitamente, usar ese valor
    # 2. Si no, verificar variable de entorno ATHERIA_NO_FRONTEND
    # 3. Por defecto, servir frontend (True)
    if serve_frontend is None:
        import os
        serve_frontend = os.environ.get('ATHERIA_NO_FRONTEND', '').lower() not in ('1', 'true', 'yes')
    
    # Configurar middleware para manejar proxies reversos (como Lightning AI)
    # Esto permite que el servidor funcione correctamente detrás de un proxy
    @web.middleware
    async def proxy_middleware(request, handler):
        # Logging útil para debugging en entornos con proxy
        if request.path == '/ws':
            forwarded_for = request.headers.get('X-Forwarded-For', 'N/A')
            forwarded_proto = request.headers.get('X-Forwarded-Proto', 'N/A')
            logging.debug(f"WebSocket connection attempt - X-Forwarded-For: {forwarded_for}, X-Forwarded-Proto: {forwarded_proto}")
        return await handler(request)
    
    app.middlewares.append(proxy_middleware)
    
    setup_routes(app, serve_frontend=serve_frontend)
    
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, global_cfg.LAB_SERVER_HOST, global_cfg.LAB_SERVER_PORT)
    
    logging.info(f"Servidor Aetheria listo y escuchando en http://{global_cfg.LAB_SERVER_HOST}:{global_cfg.LAB_SERVER_PORT}")
    logging.info("Nota: Si estás usando Lightning AI o un proxy reverso, asegúrate de que el puerto esté correctamente exportado.")
    
    # Iniciar el sitio en una tarea separada para poder detenerlo cuando sea necesario
    await site.start()
    
    # Mantiene el servidor corriendo hasta que se solicite el shutdown
    if shutdown_event:
        try:
            await shutdown_event.wait()
            logging.info("Shutdown solicitado. Cerrando servidor...")
        except asyncio.CancelledError:
            logging.info("Shutdown cancelado.")
        finally:
            # Detener el sitio y limpiar el runner con timeout más corto y forzado
            try:
                logging.info("Deteniendo sitio...")
                # Reducir timeout a 2 segundos
                await asyncio.wait_for(site.stop(), timeout=2.0)
                logging.info("Sitio detenido correctamente.")
            except asyncio.TimeoutError:
                logging.warning("Timeout al detener el sitio. Forzando cierre...")
                # Si hay un servidor subyacente, intentar cerrarlo directamente
                if hasattr(site, '_server') and site._server:
                    site._server.close()
            except Exception as e:
                logging.warning(f"Error al detener el sitio: {e}")
            
            try:
                logging.info("Limpiando runner...")
                # Reducir timeout a 2 segundos
                await asyncio.wait_for(runner.cleanup(), timeout=2.0)
                logging.info("Runner limpiado. Servidor cerrado correctamente.")
            except asyncio.TimeoutError:
                logging.warning("Timeout al limpiar el runner. Forzando cierre...")
                # Forzar limpieza cancelando todas las tareas pendientes
                for task in asyncio.all_tasks():
                    if not task.done():
                        task.cancel()
            except Exception as e:
                logging.warning(f"Error al limpiar el runner: {e}")
    else:
        # Fallback: mantener el servidor corriendo indefinidamente
        await asyncio.Event().wait()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
