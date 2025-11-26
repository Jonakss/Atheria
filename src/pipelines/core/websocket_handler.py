"""Handler de WebSocket para el pipeline server."""
import asyncio
import json
import logging
import uuid

from aiohttp import web

from src.server.server_state import g_state, send_notification
from ...utils import get_experiment_list


logger = logging.getLogger(__name__)


async def websocket_handler(request, handlers):
    """
    Maneja las conexiones WebSocket entrantes.
    
    Args:
        request: Request HTTP de aiohttp
        handlers: Diccionario de handlers para comandos WebSocket
    """
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
        from ...engines.qca_engine import Aetheria_Motor
        python_version = getattr(Aetheria_Motor, 'VERSION', None) or getattr(Aetheria_Motor, 'get_version', lambda: "unknown")() if hasattr(Aetheria_Motor, 'get_version') else "unknown"
        
        native_version = None
        wrapper_version = None
        try:
            import atheria_core
            # Obtener versión del motor nativo si está disponible
            from ...engines.native_engine_wrapper import NativeEngineWrapper
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
                    
                    logging.info(f"📥 Comando recibido: {scope}.{command} de [{ws_id}]")
                    logging.debug(f"📥 Args del comando: {args}")
                    
                    # Buscar y ejecutar el handler correspondiente
                    if scope in handlers and command in handlers[scope]:
                        handler = handlers[scope][command]
                        logging.info(f"✅ Handler encontrado para {scope}.{command}, ejecutando...")
                        try:
                            # CRÍTICO: Agregar timeout global a handlers para prevenir bloqueos infinitos
                            # Algunos handlers pueden bloquearse (especialmente conversiones denso → disperso)
                            try:
                                await asyncio.wait_for(handler(args), timeout=30.0)  # 30 segundos máximo por comando
                                logging.info(f"✅ Handler {scope}.{command} completado exitosamente")
                            except asyncio.TimeoutError:
                                logging.error(f"❌ Handler {scope}.{command} excedió timeout de 30s. Posible bloqueo.")
                                try:
                                    await send_notification(ws, f"Error: {scope}.{command} excedió tiempo límite. Intenta de nuevo.", "error")
                                except (ConnectionResetError, ConnectionError, OSError, RuntimeError):
                                    break
                            except (ConnectionResetError, ConnectionError, OSError, RuntimeError) as handler_error:
                                # Cliente se desconectó durante el handler - normal
                                logging.debug(f"Cliente desconectado durante handler {scope}.{command}: {type(handler_error).__name__}")
                                break
                            except Exception as handler_error:
                                # Error inesperado en el handler - loguear pero no crashear
                                logging.error(f"Error ejecutando handler {scope}.{command}: {handler_error}", exc_info=True)
                                try:
                                    await send_notification(ws, f"Error ejecutando {scope}.{command}: {str(handler_error)[:100]}", "error")
                                except (ConnectionResetError, ConnectionError, OSError, RuntimeError):
                                    break  # Cliente desconectado, salir del loop
                        except Exception as outer_error:
                            # Error incluso antes de ejecutar handler (timeout, etc.)
                            logging.error(f"Error preparando handler {scope}.{command}: {outer_error}", exc_info=True)
                    else:
                        logging.warning(f"⚠️ Comando desconocido: {scope}.{command}")
                        logging.warning(f"⚠️ Handlers disponibles en scope '{scope}': {list(handlers.get(scope, {}).keys())}")
                        try:
                            await send_notification(ws, f"Comando desconocido: {scope}.{command}", "error")
                        except (ConnectionResetError, ConnectionError, OSError, RuntimeError):
                            pass  # Cliente desconectado, ignorar
                        
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

