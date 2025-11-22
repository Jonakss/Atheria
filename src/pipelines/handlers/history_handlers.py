"""Handlers para comandos de historial (Guardar, Cargar, Listar, Limpiar)."""
import logging
import json
from pathlib import Path

from ...server.server_state import g_state, broadcast, send_notification, send_to_websocket
from ...managers.history_manager import HISTORY_DIR

logger = logging.getLogger(__name__)


async def handle_enable_history(args):
    """Habilita o deshabilita el guardado de historia de simulación."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    try:
        enabled = args.get('enabled', True)
        g_state['history_enabled'] = enabled
        
        msg = f"✅ Guardado de historia {'habilitado' if enabled else 'deshabilitado'}"
        logging.info(msg)
        if ws:
            await send_notification(ws, msg, "success")
        
    except Exception as e:
        logging.error(f"Error al configurar historia: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al configurar historia: {str(e)}", "error")


async def handle_save_history(args):
    """Guarda el historial de simulación a un archivo."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    try:
        history = g_state.get('simulation_history')
        if not history:
            msg = "⚠️ No hay historial disponible."
            if ws:
                await send_notification(ws, msg, "warning")
            return
        
        filename = args.get('filename')
        filepath = history.save_to_file(filename)
        
        msg = f"✅ Historial guardado: {filepath.name} ({len(history.frames)} frames)"
        logging.info(msg)
        if ws:
            await send_notification(ws, msg, "success")
            await send_to_websocket(ws, "history_saved", {
                "filename": filepath.name,
                "filepath": str(filepath),
                "frames": len(history.frames)
            })
        
    except Exception as e:
        logging.error(f"Error al guardar historial: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al guardar historial: {str(e)}", "error")


async def handle_clear_history(args):
    """Limpia el historial de simulación."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    try:
        history = g_state.get('simulation_history')
        if history:
            n_frames = len(history.frames)
            history.clear()
            msg = f"✅ Historial limpiado ({n_frames} frames eliminados)"
            logging.info(msg)
            if ws:
                await send_notification(ws, msg, "success")
        else:
            msg = "⚠️ No hay historial para limpiar."
            if ws:
                await send_notification(ws, msg, "warning")
        
    except Exception as e:
        logging.error(f"Error al limpiar historial: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al limpiar historial: {str(e)}", "error")


async def handle_list_history_files(args):
    """Lista los archivos de historia guardados."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    try:
        history_files = []
        if HISTORY_DIR.exists():
            for filepath in HISTORY_DIR.glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    metadata = data.get('metadata', {})
                    frames = data.get('frames', [])
                    
                    if frames:
                        steps = [f.get('step', 0) for f in frames]
                        history_files.append({
                            'filename': filepath.name,
                            'filepath': str(filepath),
                            'frames': len(frames),
                            'created_at': metadata.get('created_at'),
                            'min_step': min(steps) if steps else None,
                            'max_step': max(steps) if steps else None
                        })
                except Exception as e:
                    logging.warning(f"Error leyendo archivo de historia {filepath}: {e}")
        
        # Ordenar por fecha de creación (más reciente primero)
        history_files.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        if ws:
            await send_to_websocket(ws, "history_files_list", {
                "files": history_files
            })
        
    except Exception as e:
        logging.error(f"Error listando archivos de historia: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error listando archivos: {str(e)}", "error")


async def handle_load_history_file(args):
    """Carga un archivo de historia."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    try:
        filename = args.get('filename')
        if not filename:
            if ws:
                await send_notification(ws, "⚠️ Nombre de archivo no proporcionado.", "warning")
            return
        
        filepath = HISTORY_DIR / filename
        if not filepath.exists():
            if ws:
                await send_notification(ws, f"⚠️ Archivo no encontrado: {filename}", "warning")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        
        if ws:
            await send_to_websocket(ws, "history_file_loaded", {
                "filename": filename,
                "frames": frames,
                "metadata": data.get('metadata', {})
            })
            await send_notification(ws, f"✅ Historia cargada: {len(frames)} frames", "success")
        
    except Exception as e:
        logging.error(f"Error cargando archivo de historia: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error cargando archivo: {str(e)}", "error")


HANDLERS = {
    "enable_history": handle_enable_history,
    "save_history": handle_save_history,
    "clear_history": handle_clear_history,
    "list_history_files": handle_list_history_files,
    "load_history_file": handle_load_history_file
}
