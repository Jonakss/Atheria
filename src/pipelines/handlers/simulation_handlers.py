"""Handlers para comandos de simulación (configuración, velocidad, live feed, etc.)."""
import asyncio
import logging

from ...server.server_state import g_state, broadcast, send_notification, optimize_frame_payload
from ..viz import get_visualization_data

logger = logging.getLogger(__name__)


async def handle_set_viz(args):
    """Cambia el tipo de visualización."""
    viz_type = args.get("viz_type", "density")
    g_state['viz_type'] = viz_type
    if (ws := g_state['websockets'].get(args.get('ws_id'))):
        await send_notification(ws, f"Visualización cambiada a: {viz_type}", "info")
    # Si hay un motor activo, enviar un frame actualizado inmediatamente
    # SOLO si live_feed está habilitado
    live_feed_enabled = g_state.get('live_feed_enabled', True)
    if live_feed_enabled:
        motor = g_state.get('motor')
        if motor and hasattr(motor, 'state') and motor.state and hasattr(motor.state, 'psi') and motor.state.psi is not None:
            try:
                delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                viz_data = get_visualization_data(
                    motor.state.psi, 
                    viz_type,
                    delta_psi=delta_psi,
                    motor=motor
                )
                
                if viz_data and isinstance(viz_data, dict):
                    map_data = viz_data.get("map_data", [])
                    if map_data and len(map_data) > 0:
                        compression_enabled = g_state.get('data_compression_enabled', True)
                        downsample_factor = g_state.get('downsample_factor', 1)
                        
                        frame_payload_raw = {
                            "step": g_state.get('simulation_step', 0),
                            "timestamp": asyncio.get_event_loop().time(),
                            "map_data": map_data,
                            "hist_data": viz_data.get("hist_data", {}),
                            "poincare_coords": viz_data.get("poincare_coords", []),
                            "phase_attractor": viz_data.get("phase_attractor"),
                            "flow_data": viz_data.get("flow_data"),
                            "phase_hsv_data": viz_data.get("phase_hsv_data"),
                            "complex_3d_data": viz_data.get("complex_3d_data"),
                            "simulation_info": {
                                "step": g_state.get('simulation_step', 0),
                                "is_paused": g_state.get('is_paused', True),
                                "live_feed_enabled": live_feed_enabled,
                                "fps": g_state.get('current_fps', 0.0)
                            }
                        }
                        
                        if compression_enabled or downsample_factor > 1:
                            frame_payload = await optimize_frame_payload(
                                frame_payload_raw,
                                enable_compression=compression_enabled,
                                downsample_factor=downsample_factor,
                                viz_type=viz_type
                            )
                        else:
                            frame_payload = frame_payload_raw
                        
                        await broadcast({"type": "simulation_frame", "payload": frame_payload})
            except Exception as e:
                logging.error(f"Error al actualizar visualización: {e}", exc_info=True)
                if ws: await send_notification(ws, f"Error al actualizar visualización: {str(e)}", "error")


async def handle_update_visualization(args):
    """Actualiza manualmente la visualización (útil para modo no-live)."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    motor = g_state.get('motor')
    
    if not motor:
        if ws: await send_notification(ws, "⚠️ No hay modelo cargado.", "warning")
        return
    
    if not motor.state or motor.state.psi is None:
        if ws: await send_notification(ws, "⚠️ El modelo no tiene un estado válido.", "warning")
        return
    
    try:
        delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
        viz_type = g_state.get('viz_type', 'density')
        viz_data = get_visualization_data(
            motor.state.psi, 
            viz_type,
            delta_psi=delta_psi,
            motor=motor
        )
        
        if viz_data and isinstance(viz_data, dict):
            map_data = viz_data.get("map_data", [])
            if map_data and len(map_data) > 0:
                compression_enabled = g_state.get('data_compression_enabled', True)
                downsample_factor = g_state.get('downsample_factor', 1)
                
                frame_payload_raw = {
                    "step": g_state.get('simulation_step', 0),
                    "timestamp": asyncio.get_event_loop().time(),
                    "map_data": map_data,
                    "hist_data": viz_data.get("hist_data", {}),
                    "poincare_coords": viz_data.get("poincare_coords", []),
                    "phase_attractor": viz_data.get("phase_attractor"),
                    "flow_data": viz_data.get("flow_data"),
                    "phase_hsv_data": viz_data.get("phase_hsv_data"),
                    "complex_3d_data": viz_data.get("complex_3d_data"),
                    "simulation_info": {
                        "step": g_state.get('simulation_step', 0),
                        "is_paused": g_state.get('is_paused', True),
                        "live_feed_enabled": g_state.get('live_feed_enabled', True),
                        "fps": g_state.get('current_fps', 0.0)
                    }
                }
                
                if compression_enabled or downsample_factor > 1:
                    frame_payload = await optimize_frame_payload(
                        frame_payload_raw,
                        enable_compression=compression_enabled,
                        downsample_factor=downsample_factor,
                        viz_type=viz_type
                    )
                else:
                    frame_payload = frame_payload_raw
                
                await broadcast({"type": "simulation_frame", "payload": frame_payload})
                if ws: await send_notification(ws, "✅ Visualización actualizada.", "success")
    except Exception as e:
        logging.error(f"Error al actualizar visualización: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al actualizar visualización: {str(e)}", "error")


async def handle_set_simulation_speed(args):
    """Controla la velocidad de la simulación (multiplicador)."""
    speed = args.get("speed", 1.0)
    if speed < 0.1:
        speed = 0.1
    elif speed > 100.0:
        speed = 100.0
    
    g_state['simulation_speed'] = float(speed)
    logging.info(f"Velocidad de simulación ajustada a: {speed}x")
    
    # Enviar actualización a clientes
    await broadcast({
        "type": "simulation_speed_update",
        "payload": {"speed": g_state['simulation_speed']}
    })


async def handle_set_fps(args):
    """Controla los FPS objetivo de la simulación."""
    fps = args.get("fps", 10.0)
    if fps < 0.1:
        fps = 0.1
    elif fps > 120.0:
        fps = 120.0
    
    g_state['target_fps'] = float(fps)
    logging.info(f"FPS objetivo ajustado a: {fps}")
    
    # Enviar actualización a clientes
    await broadcast({
        "type": "simulation_fps_update",
        "payload": {"fps": g_state['target_fps']}
    })


async def handle_set_frame_skip(args):
    """Controla cuántos frames saltar para acelerar (0 = todos, 1 = cada otro, etc.)."""
    skip = args.get("skip", 0)
    if skip < 0:
        skip = 0
    elif skip > 10:
        skip = 10
    
    g_state['frame_skip'] = int(skip)
    logging.info(f"Frame skip ajustado a: {skip} (cada {skip + 1} frames se renderiza)")
    
    # Enviar actualización a clientes
    await broadcast({
        "type": "simulation_frame_skip_update",
        "payload": {"skip": g_state['frame_skip']}
    })


async def handle_set_live_feed(args):
    """Activa o desactiva el live feed (envío automático de frames)."""
    enabled = args.get("enabled", True)
    g_state['live_feed_enabled'] = enabled
    
    ws = g_state['websockets'].get(args.get('ws_id'))
    if enabled:
        logging.info("Live feed activado. La simulación continuará ejecutándose y enviando datos en tiempo real.")
        if ws: await send_notification(ws, "Live feed activado.", "info")
    else:
        logging.info("Live feed desactivado. La simulación continuará ejecutándose pero sin enviar frames automáticamente.")
        if ws: await send_notification(ws, "Live feed desactivado. Usa 'Actualizar Visualización' para ver cambios.", "info")
    
    # Enviar actualización a todos los clientes
    await broadcast({
        "type": "live_feed_status",
        "payload": {"enabled": enabled}
    })


async def handle_set_steps_interval(args):
    """
    Configura el intervalo de pasos para el envío de frames cuando live_feed está DESACTIVADO.
    
    Args:
        interval o steps_interval: Intervalo en pasos. Valores:
            - -1: Fullspeed (no enviar frames automáticamente, máxima velocidad)
            - 0: Modo manual (solo actualizar con botón)
            - 1-1000000: Enviar frame cada N pasos automáticamente
    """
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    # Aceptar 'interval' o 'steps_interval' como parámetro (compatibilidad)
    interval = args.get("interval") or args.get("steps_interval", 10)
    
    try:
        interval = int(interval)
        
        if interval == -1:
            # Modo fullspeed: no enviar frames automáticamente
            g_state['steps_interval'] = -1
            g_state['steps_interval_counter'] = 0
            msg = "✅ Modo fullspeed activado: simulación a máxima velocidad sin enviar frames"
        elif interval < 0:
            # Cualquier valor negativo se convierte a -1 (fullspeed)
            interval = -1
            g_state['steps_interval'] = -1
            g_state['steps_interval_counter'] = 0
            msg = "✅ Modo fullspeed activado: simulación a máxima velocidad sin enviar frames"
        elif interval == 0:
            # Modo manual: solo actualizar con botón
            g_state['steps_interval'] = 0
            g_state['steps_interval_counter'] = 0
            msg = "✅ Modo manual activado: solo actualizar con botón 'Actualizar Visualización'"
        elif interval > 1_000_000:
            # Limitar a 1 millón
            interval = 1_000_000
            logging.warning(f"⚠️ steps_interval limitado a 1,000,000 (valor solicitado: {args.get('interval') or args.get('steps_interval')})")
            g_state['steps_interval'] = interval
            g_state['steps_interval_counter'] = 0
            steps_str = f"{interval:,}".replace(",", ".")
            msg = f"✅ Intervalo de pasos configurado: cada {steps_str} pasos"
        else:
            g_state['steps_interval'] = interval
            g_state['steps_interval_counter'] = 0
            # Formatear número con separadores de miles para mejor legibilidad
            steps_str = f"{interval:,}".replace(",", ".")
            msg = f"✅ Intervalo de pasos configurado: cada {steps_str} pasos"
        
        logging.info(msg)
        if ws:
            await send_notification(ws, msg, "success")
    except (ValueError, TypeError) as e:
        logging.error(f"Error al configurar intervalo de pasos: {e}", exc_info=True)
        g_state['steps_interval'] = 10  # Valor por defecto seguro
        if ws:
            await send_notification(ws, f"Error al configurar intervalo: {str(e)}. Usando valor por defecto: 10", "error")


# Más handlers de simulación se agregarán aquí según se extraigan de pipeline_server.py
# Por ahora, los handlers complejos (snapshots, history, etc.) se mantienen en pipeline_server.py

HANDLERS = {
    "set_viz": handle_set_viz,
    "update_visualization": handle_update_visualization,
    "set_speed": handle_set_simulation_speed,
    "set_fps": handle_set_fps,
    "set_frame_skip": handle_set_frame_skip,
    "set_live_feed": handle_set_live_feed,
    "set_steps_interval": handle_set_steps_interval,
}

