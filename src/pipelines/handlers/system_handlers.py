"""Handlers para comandos del sistema (shutdown, refresh, etc.)."""
import logging

from ...server.server_state import g_state, send_notification, send_to_websocket
from ...utils import get_experiment_list

logger = logging.getLogger(__name__)


async def handle_shutdown(args):
    """Apaga el servidor desde la UI."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    logging.info(f"Recibida orden de apagar servidor desde [{args.get('ws_id')}]")
    
    if ws:
        try:
            await send_notification(ws, "🛑 Servidor apagándose...", "info")
            await send_to_websocket(ws, "server_shutting_down", {})
        except (ConnectionResetError, ConnectionError, OSError, RuntimeError):
            pass  # Cliente ya desconectado
    
    # Limpiar recursos antes de apagar
    try:
        motor = g_state.get('motor')
        if motor and hasattr(motor, 'cleanup'):
            try:
                motor.cleanup()
            except Exception as cleanup_error:
                logging.warning(f"Error durante cleanup antes de apagar: {cleanup_error}")
    except Exception as e:
        logging.warning(f"Error limpiando recursos antes de apagar: {e}")
    
    # Enviar mensaje de cierre a todos los clientes
    try:
        from ...server.server_state import broadcast
        await broadcast({
            "type": "server_shutdown",
            "payload": {"message": "Servidor apagándose..."}
        })
    except Exception as e:
        logging.warning(f"Error enviando mensaje de cierre: {e}")
    
    # Importar aquí para evitar import circular
    import asyncio
    import sys
    
    # Dar tiempo para que los mensajes se envíen
    await asyncio.sleep(0.5)
    
    # Apagar el servidor
    logging.info("🛑 Apagando servidor...")
    sys.exit(0)


async def handle_refresh_experiments(args):
    """Actualiza la lista de experimentos disponibles."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    try:
        experiments = get_experiment_list()
        if ws:
            await send_to_websocket(ws, "experiments_list", {"experiments": experiments})
            await send_notification(ws, f"✅ Lista de experimentos actualizada ({len(experiments)} encontrados)", "success")
        logging.info(f"Lista de experimentos actualizada: {len(experiments)} experimentos")
    except Exception as e:
        logging.error(f"Error actualizando lista de experimentos: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error actualizando lista: {str(e)}", "error")


async def handle_toggle_logs(args):
    """Activa o desactiva el streaming de logs."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    enabled = args.get('enabled', True)
    
    # Buscar el handler de logs en el root logger
    root_logger = logging.getLogger()
    log_handler = None
    for handler in root_logger.handlers:
        if hasattr(handler, 'enabled') and handler.__class__.__name__ == 'WebSocketLogHandler':
            log_handler = handler
            break
    
    if log_handler:
        log_handler.enabled = enabled
        status = "activados" if enabled else "desactivados"
        logging.info(f"Logs de WebSocket {status}")
        if ws:
            await send_notification(ws, f"Logs {status}", "info")
    else:
        logging.warning("No se encontró WebSocketLogHandler para configurar")
        if ws:
            await send_notification(ws, "Error: No se encontró el sistema de logs", "error")


HANDLERS = {
    "shutdown": handle_shutdown,
    "refresh_experiments": handle_refresh_experiments,
    "toggle_logs": handle_toggle_logs,
}

