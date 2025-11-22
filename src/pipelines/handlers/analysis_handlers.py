"""Handlers para comandos de análisis (Atlas del Universo, Química Celular)."""
import asyncio
import logging
import threading
import concurrent.futures

from ...server.server_state import g_state, broadcast, send_notification, send_to_websocket
from ...analysis.analysis import analyze_universe_atlas, analyze_cell_chemistry, calculate_phase_map_metrics

logger = logging.getLogger(__name__)


async def handle_analyze_universe_atlas(args):
    """
    Crea un "Atlas del Universo" analizando la evolución temporal usando t-SNE.
    """
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    try:
        # Cancelar análisis anterior si hay uno corriendo
        if g_state.get('analysis_status') == 'running':
            logging.info("Cancelando análisis anterior...")
            if g_state.get('analysis_task'):
                g_state['analysis_task'].cancel()
            if g_state.get('analysis_cancel_event'):
                g_state['analysis_cancel_event'].set()
            g_state['analysis_status'] = 'idle'
            g_state['analysis_type'] = None
            await broadcast({
                "type": "analysis_status_update",
                "payload": {"status": "cancelled", "type": None}
            })
        
        # Establecer estado de análisis
        g_state['analysis_status'] = 'running'
        g_state['analysis_type'] = 'universe_atlas'
        g_state['analysis_cancel_event'] = threading.Event()
        
        # Notificar inicio
        await broadcast({
            "type": "analysis_status_update",
            "payload": {"status": "running", "type": "universe_atlas"}
        })
        
        if ws:
            await send_notification(ws, "🔄 Analizando Atlas del Universo...", "info")
        
        # Habilitar snapshots automáticamente si no están habilitados
        if not g_state.get('snapshot_enabled', False):
            g_state['snapshot_enabled'] = True
            logging.info("Snapshots habilitados automáticamente para análisis")
            if ws:
                await send_notification(ws, "📸 Captura de snapshots habilitada automáticamente para análisis", "info")
        
        # Obtener snapshots almacenados
        snapshots = g_state.get('snapshots', [])
        
        if len(snapshots) < 2:
            msg = f"⚠️ Se necesitan al menos 2 snapshots para el análisis. Actualmente hay {len(snapshots)}. Ejecuta la simulación durante más tiempo para capturar snapshots (cada {g_state.get('snapshot_interval', 500)} pasos)."
            logging.warning(msg)
            g_state['analysis_status'] = 'idle'
            g_state['analysis_type'] = None
            await broadcast({
                "type": "analysis_status_update",
                "payload": {"status": "idle", "type": None}
            })
            if ws:
                await send_notification(ws, msg, "warning")
                await send_to_websocket(ws, "analysis_universe_atlas", {
                    "error": msg,
                    "n_snapshots": len(snapshots),
                    "snapshot_interval": g_state.get('snapshot_interval', 500)
                })
            return
        
        # Extraer tensores psi de los snapshots
        psi_snapshots = [snapshot['psi'] for snapshot in snapshots]
        
        # Obtener parámetros de análisis (con valores por defecto)
        compression_dim = args.get('compression_dim', 64)
        perplexity = args.get('perplexity', 30)
        n_iter = args.get('n_iter', 1000)
        
        logging.info(f"Iniciando análisis Atlas del Universo con {len(psi_snapshots)} snapshots...")
        
        # Ejecutar análisis en un thread separado para no bloquear el event loop
        loop = asyncio.get_event_loop()
        
        # Crear tarea de análisis
        async def run_analysis():
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        analyze_universe_atlas,
                        psi_snapshots,
                        compression_dim,
                        perplexity,
                        n_iter
                    )
                
                    # Verificar si fue cancelado
                    if g_state.get('analysis_cancel_event') and g_state['analysis_cancel_event'].is_set():
                        logging.info("Análisis cancelado por el usuario")
                        g_state['analysis_status'] = 'idle'
                        g_state['analysis_type'] = None
                        await broadcast({
                            "type": "analysis_status_update",
                            "payload": {"status": "cancelled", "type": None}
                        })
                        return
                    
                    # Calcular métricas
                    metrics = calculate_phase_map_metrics(result['coords'])
                    result['metrics'] = metrics
                    
                    logging.info(f"Análisis Atlas del Universo completado: {len(result['coords'])} puntos, spread={metrics['spread']:.2f}")
                    
                    g_state['analysis_status'] = 'idle'
                    g_state['analysis_type'] = None
                    await broadcast({
                        "type": "analysis_status_update",
                        "payload": {"status": "completed", "type": None}
                    })
                    
                    if ws:
                        await send_notification(ws, f"✅ Atlas del Universo completado ({len(result['coords'])} puntos)", "success")
                        await send_to_websocket(ws, "analysis_universe_atlas", result)
            except asyncio.CancelledError:
                logging.info("Análisis cancelado")
                g_state['analysis_status'] = 'idle'
                g_state['analysis_type'] = None
                await broadcast({
                    "type": "analysis_status_update",
                    "payload": {"status": "cancelled", "type": None}
                })
            except Exception as e:
                logging.error(f"Error en análisis: {e}", exc_info=True)
                g_state['analysis_status'] = 'idle'
                g_state['analysis_type'] = None
                await broadcast({
                    "type": "analysis_status_update",
                    "payload": {"status": "error", "type": None, "error": str(e)}
                })
                if ws:
                    await send_notification(ws, f"❌ Error en análisis: {str(e)}", "error")
                    await send_to_websocket(ws, "analysis_universe_atlas", {
                        "error": str(e)
                    })
        
        task = asyncio.create_task(run_analysis())
        g_state['analysis_task'] = task
        
    except Exception as e:
        logging.error(f"Error en análisis Atlas del Universo: {e}", exc_info=True)
        g_state['analysis_status'] = 'idle'
        g_state['analysis_type'] = None
        await broadcast({
            "type": "analysis_status_update",
            "payload": {"status": "error", "type": None, "error": str(e)}
        })
        if ws:
            await send_notification(ws, f"❌ Error en análisis: {str(e)}", "error")
            await send_to_websocket(ws, "analysis_universe_atlas", {
                "error": str(e)
            })


async def handle_analyze_cell_chemistry(args):
    """
    Crea un "Mapa Químico" analizando los tipos de células en el estado actual usando t-SNE.
    """
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    try:
        # Cancelar análisis anterior si hay uno corriendo
        if g_state.get('analysis_status') == 'running':
            logging.info("Cancelando análisis anterior...")
            if g_state.get('analysis_task'):
                g_state['analysis_task'].cancel()
            if g_state.get('analysis_cancel_event'):
                g_state['analysis_cancel_event'].set()
            g_state['analysis_status'] = 'idle'
            g_state['analysis_type'] = None
            await broadcast({
                "type": "analysis_status_update",
                "payload": {"status": "cancelled", "type": None}
            })
        
        # Establecer estado de análisis
        g_state['analysis_status'] = 'running'
        g_state['analysis_type'] = 'cell_chemistry'
        g_state['analysis_cancel_event'] = threading.Event()
        
        # Notificar inicio
        await broadcast({
            "type": "analysis_status_update",
            "payload": {"status": "running", "type": "cell_chemistry"}
        })
        
        if ws:
            await send_notification(ws, "🔄 Analizando Mapa Químico...", "info")
        
        # Obtener estado actual del motor
        motor = g_state.get('motor')
        if not motor or not motor.state or motor.state.psi is None:
            msg = "⚠️ No hay simulación activa. Carga un experimento y ejecuta la simulación primero."
            logging.warning(msg)
            g_state['analysis_status'] = 'idle'
            g_state['analysis_type'] = None
            await broadcast({
                "type": "analysis_status_update",
                "payload": {"status": "idle", "type": None}
            })
            if ws:
                await send_notification(ws, msg, "warning")
                await send_to_websocket(ws, "analysis_cell_chemistry", {
                    "error": msg
                })
            return
        
        psi = motor.state.psi
        
        # Obtener parámetros de análisis (con valores por defecto)
        n_samples = args.get('n_samples', 10000)
        perplexity = args.get('perplexity', 30)
        n_iter = args.get('n_iter', 1000)
        
        logging.info(f"Iniciando análisis Mapa Químico...")
        
        # Ejecutar análisis en un thread separado para no bloquear el event loop
        loop = asyncio.get_event_loop()
        
        # Crear tarea de análisis
        async def run_analysis():
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        analyze_cell_chemistry,
                        psi,
                        n_samples,
                        perplexity,
                        n_iter
                    )
                
                    # Verificar si fue cancelado
                    if g_state.get('analysis_cancel_event') and g_state['analysis_cancel_event'].is_set():
                        logging.info("Análisis cancelado por el usuario")
                        g_state['analysis_status'] = 'idle'
                        g_state['analysis_type'] = None
                        await broadcast({
                            "type": "analysis_status_update",
                            "payload": {"status": "cancelled", "type": None}
                        })
                        return
                    
                    logging.info(f"Análisis Mapa Químico completado: {len(result['coords'])} células")
                    
                    g_state['analysis_status'] = 'idle'
                    g_state['analysis_type'] = None
                    await broadcast({
                        "type": "analysis_status_update",
                        "payload": {"status": "completed", "type": None}
                    })
                    
                    if ws:
                        await send_notification(ws, f"✅ Mapa Químico completado ({len(result['coords'])} células)", "success")
                        await send_to_websocket(ws, "analysis_cell_chemistry", result)
            except asyncio.CancelledError:
                logging.info("Análisis cancelado")
                g_state['analysis_status'] = 'idle'
                g_state['analysis_type'] = None
                await broadcast({
                    "type": "analysis_status_update",
                    "payload": {"status": "cancelled", "type": None}
                })
            except Exception as e:
                logging.error(f"Error en análisis: {e}", exc_info=True)
                g_state['analysis_status'] = 'idle'
                g_state['analysis_type'] = None
                await broadcast({
                    "type": "analysis_status_update",
                    "payload": {"status": "error", "type": None, "error": str(e)}
                })
                if ws:
                    await send_notification(ws, f"❌ Error en análisis: {str(e)}", "error")
                    await send_to_websocket(ws, "analysis_cell_chemistry", {
                        "error": str(e)
                    })
        
        task = asyncio.create_task(run_analysis())
        g_state['analysis_task'] = task
        
    except Exception as e:
        logging.error(f"Error en análisis Mapa Químico: {e}", exc_info=True)
        g_state['analysis_status'] = 'idle'
        g_state['analysis_type'] = None
        await broadcast({
            "type": "analysis_status_update",
            "payload": {"status": "error", "type": None, "error": str(e)}
        })
        if ws:
            await send_notification(ws, f"❌ Error en análisis: {str(e)}", "error")
            await send_to_websocket(ws, "analysis_cell_chemistry", {
                "error": str(e)
            })


async def handle_cancel_analysis(args):
    """Cancela cualquier análisis en curso."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    if g_state.get('analysis_status') == 'running':
        logging.info("Solicitud de cancelación de análisis recibida")
        
        # Señalizar cancelación
        if g_state.get('analysis_cancel_event'):
            g_state['analysis_cancel_event'].set()
        
        # Cancelar tarea asyncio si existe
        if g_state.get('analysis_task'):
            g_state['analysis_task'].cancel()
        
        g_state['analysis_status'] = 'idle'
        g_state['analysis_type'] = None
        
        await broadcast({
            "type": "analysis_status_update",
            "payload": {"status": "cancelled", "type": None}
        })
        
        if ws:
            await send_notification(ws, "🛑 Análisis cancelado.", "info")
    else:
        if ws:
            await send_notification(ws, "⚠️ No hay análisis en curso para cancelar.", "warning")


async def handle_clear_snapshots(args):
    """Limpia todos los snapshots almacenados."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    if 'snapshots' in g_state:
        count = len(g_state['snapshots'])
        g_state['snapshots'] = []
        
        # Forzar garbage collection
        import gc
        gc.collect()
        
        msg = f"✅ {count} snapshots eliminados y memoria liberada."
        logging.info(msg)
        if ws:
            await send_notification(ws, msg, "success")
            await send_to_websocket(ws, "snapshot_count", {
                "count": 0,
                "step": g_state.get('simulation_step', 0)
            })
    else:
        if ws:
            await send_notification(ws, "⚠️ No hay snapshots para limpiar.", "warning")


HANDLERS = {
    "universe_atlas": handle_analyze_universe_atlas,
    "cell_chemistry": handle_analyze_cell_chemistry,
    "cancel": handle_cancel_analysis,
    "clear_snapshots": handle_clear_snapshots
}
