"""Handlers para comandos de inferencia (play, pause, load, reset, etc.)."""
import asyncio
import logging
import torch
import numpy as np

from ...server.server_state import g_state, broadcast, send_notification
from ..core.status_helpers import build_inference_status_payload

logger = logging.getLogger(__name__)


async def handle_play(args):
    """Inicia la simulación."""
    logging.info("🎮 handle_play() llamado - Iniciando simulación...")
    ws = g_state['websockets'].get(args['ws_id'])
    
    # Validar que haya un motor cargado antes de iniciar
    motor = g_state.get('motor')
    if not motor:
        msg = "⚠️ No hay un modelo cargado. Primero debes cargar un experimento entrenado."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    # Validar que el motor tenga estado válido
    motor_is_native = g_state.get('motor_is_native', False)
    
    if motor_is_native and hasattr(motor, 'native_engine'):
        # Motor nativo: verificar que tenga partículas o estado inicializado
        # El motor nativo usa lazy conversion, así que no verificamos motor.state.psi aquí
        # En su lugar, verificamos que el motor tenga partículas
        try:
            # Verificar que el motor nativo esté inicializado
            if not hasattr(motor, 'model_loaded') or not motor.model_loaded:
                msg = "⚠️ El motor nativo no tiene un modelo cargado. Intenta recargar el experimento."
                logging.warning(msg)
                if ws: await send_notification(ws, msg, "warning")
                return
            
            # Intentar obtener el estado denso para verificar que haya datos
            # Esto forzará la conversión si es necesario
            import torch
            psi = motor.get_dense_state(check_pause_callback=lambda: False)
            if psi is None or (isinstance(psi, torch.Tensor) and psi.numel() == 0):
                logging.warning("⚠️ Motor nativo no tiene estado válido. Intentando inicializar...")
                # Intentar inicializar con partículas aleatorias
                grid_size = g_state.get('inference_grid_size', 256)
                # CRÍTICO: Agregar más partículas para asegurar propagación
                num_particles = max(500, (grid_size * grid_size) // 20)  # Aumentar número de partículas
                if hasattr(motor, 'add_initial_particles'):
                    motor.add_initial_particles(num_particles)
                    logging.info(f"✅ {num_particles} partículas agregadas al motor nativo (incluye paso de propagación)")
                    # add_initial_particles ahora ejecuta un paso automáticamente, pero forzamos reconversión de todas formas
                    motor._dense_state_stale = True
                    motor.state.psi = None
                    psi = motor.get_dense_state(check_pause_callback=lambda: False)
                    psi_abs_max = psi.abs().max().item() if psi is not None and isinstance(psi, torch.Tensor) else 0.0
                    if psi is None or (isinstance(psi, torch.Tensor) and psi.numel() == 0) or psi_abs_max < 1e-10:
                        # Intentar ejecutar otro paso manualmente
                        try:
                            motor.native_engine.step_native()
                            motor._dense_state_stale = True
                            motor.state.psi = None
                            psi = motor.get_dense_state(check_pause_callback=lambda: False)
                            psi_abs_max = psi.abs().max().item() if psi is not None and isinstance(psi, torch.Tensor) else 0.0
                            logging.info(f"📊 Después de paso adicional, psi max abs={psi_abs_max:.6e}")
                        except Exception as step_error:
                            logging.error(f"❌ Error ejecutando paso adicional: {step_error}", exc_info=True)
                        
                        if psi is None or (isinstance(psi, torch.Tensor) and psi.numel() == 0) or psi_abs_max < 1e-10:
                            msg = "⚠️ No se pudo inicializar el estado del motor nativo. Intenta reiniciar."
                            logging.error(msg)
                            if ws: await send_notification(ws, msg, "error")
                            return
                else:
                    msg = "⚠️ El motor nativo no tiene un estado válido y no se puede inicializar automáticamente."
                    logging.error(msg)
                    if ws: await send_notification(ws, msg, "error")
                    return
            
            # Verificar que el estado tenga valores significativos
            if isinstance(psi, torch.Tensor):
                psi_abs_max = psi.abs().max().item()
                if psi_abs_max < 1e-10:
                    logging.warning("⚠️ Estado del motor nativo tiene valores muy pequeños. Intentando inicializar...")
                    grid_size = g_state.get('inference_grid_size', 256)
                    # CRÍTICO: Agregar más partículas para asegurar propagación
                    num_particles = max(500, (grid_size * grid_size) // 20)  # Aumentar número de partículas
                    if hasattr(motor, 'add_initial_particles'):
                        motor.add_initial_particles(num_particles)
                        motor._dense_state_stale = True
                        motor.state.psi = None
                        logging.info(f"✅ {num_particles} partículas agregadas al motor nativo (incluye paso de propagación)")
                        # Verificar que se inicializó correctamente
                        psi_new = motor.get_dense_state(check_pause_callback=lambda: False)
                        psi_abs_max_new = psi_new.abs().max().item() if psi_new is not None and isinstance(psi_new, torch.Tensor) else 0.0
                        if psi_abs_max_new < 1e-10:
                            logging.warning("⚠️ psi sigue vacío después de inicialización. Ejecutando paso adicional...")
                            try:
                                motor.native_engine.step_native()
                                motor._dense_state_stale = True
                                motor.state.psi = None
                                psi_new = motor.get_dense_state(check_pause_callback=lambda: False)
                                psi_abs_max_new = psi_new.abs().max().item() if psi_new is not None and isinstance(psi_new, torch.Tensor) else 0.0
                                logging.info(f"📊 Después de paso adicional, psi max abs={psi_abs_max_new:.6e}")
                            except Exception as step_error:
                                logging.error(f"❌ Error ejecutando paso adicional: {step_error}", exc_info=True)
        except Exception as e:
            logging.error(f"❌ Error validando motor nativo: {e}", exc_info=True)
            msg = "⚠️ Error validando el estado del motor nativo. Intenta reiniciar."
            if ws: await send_notification(ws, msg, "error")
            return
    else:
        # Motor Python: verificar estado tradicional
        if not motor.state or motor.state.psi is None:
            msg = "⚠️ El modelo cargado no tiene un estado válido. Intenta reiniciar la simulación."
            logging.warning(msg)
            if ws: await send_notification(ws, msg, "warning")
            return
    
    g_state['is_paused'] = False
    
    # CRÍTICO: Enviar frame inicial si no se ha enviado todavía
    # Esto asegura que la visualización no aparezca gris
    if motor_is_native and hasattr(motor, 'get_dense_state'):
        try:
            from ..viz import get_visualization_data
            import numpy as np
            
            # Obtener estado denso
            psi = motor.get_dense_state(check_pause_callback=lambda: False)
            if psi is not None and isinstance(psi, torch.Tensor) and psi.numel() > 0:
                # Verificar que tenga valores significativos
                psi_abs_max = psi.abs().max().item()
                if psi_abs_max > 1e-10:
                    # Generar visualización
                    viz_type = g_state.get('viz_type', 'density')
                    delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                    viz_data = get_visualization_data(psi, viz_type, delta_psi=delta_psi, motor=motor)
                    
                    if viz_data and isinstance(viz_data, dict):
                        map_data = viz_data.get("map_data", [])
                        if map_data and len(map_data) > 0:
                            map_data_np = np.array(map_data) if not isinstance(map_data, np.ndarray) else map_data
                            min_val = np.min(map_data_np)
                            max_val = np.max(map_data_np)
                            range_val = max_val - min_val
                            
                            # Solo enviar si tiene rango significativo (no está todo gris)
                            if range_val > 1e-6:
                                current_step = g_state.get('simulation_step', 0)
                                frame_payload = {
                                    "step": current_step,
                                    "timestamp": asyncio.get_event_loop().time(),
                                    "map_data": map_data,
                                    "hist_data": viz_data.get("hist_data", {}),
                                    "poincare_coords": viz_data.get("poincare_coords", []),
                                    "phase_attractor": viz_data.get("phase_attractor"),
                                    "flow_data": viz_data.get("flow_data"),
                                    "phase_hsv_data": viz_data.get("phase_hsv_data"),
                                    "complex_3d_data": viz_data.get("complex_3d_data"),
                                    "simulation_info": {
                                        "step": current_step,
                                        "is_paused": False,
                                        "live_feed_enabled": g_state.get('live_feed_enabled', True),
                                        "fps": g_state.get('current_fps', 0.0)
                                    }
                                }
                                await broadcast({"type": "simulation_frame", "payload": frame_payload})
                                logging.info(f"✅ Frame inicial enviado al iniciar simulación (step={current_step}, range={range_val:.6f})")
        except Exception as e:
            logging.debug(f"Error enviando frame inicial al iniciar: {e}")
    
    logging.info(f"Simulación iniciada. Motor: {type(motor).__name__}, Step: {g_state.get('simulation_step', 0)}, Live feed: {g_state.get('live_feed_enabled', True)}")
    
    # CRÍTICO: Incluir FPS e información de estado al iniciar
    status_payload = build_inference_status_payload("running")
    status_payload.update({
        "step": g_state.get('simulation_step', 0),
        "simulation_info": {
            "step": g_state.get('simulation_step', 0),
            "is_paused": False,
            "live_feed_enabled": g_state.get('live_feed_enabled', True),
            "fps": g_state.get('current_fps', 0.0),
            "epoch": g_state.get('current_epoch', 0),
            "epoch_metrics": g_state.get('epoch_metrics', {})
        }
    })
    
    await broadcast({"type": "inference_status_update", "payload": status_payload})
    if ws: await send_notification(ws, "Simulación iniciada.", "info")


async def handle_pause(args):
    """Pausa la simulación."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    logging.info("Comando de pausa recibido. Pausando simulación...")
    g_state['is_paused'] = True
    
    # CRÍTICO: Incluir FPS actual en el estado pausado (puede ser 0 o último valor)
    # Esto permite que el frontend muestre el FPS incluso cuando está pausado
    status_payload = build_inference_status_payload("paused")
    
    # Agregar información de estado adicional
    status_payload.update({
        "step": g_state.get('simulation_step', 0),
        "simulation_info": {
            "step": g_state.get('simulation_step', 0),
            "is_paused": True,
            "live_feed_enabled": g_state.get('live_feed_enabled', True),
            "fps": g_state.get('current_fps', 0.0),  # Incluir FPS incluso cuando está pausado
            "epoch": g_state.get('current_epoch', 0),
            "epoch_metrics": g_state.get('epoch_metrics', {})
        }
    })
    
    await broadcast({"type": "inference_status_update", "payload": status_payload})
    if ws:
        await send_notification(ws, "Simulación pausada.", "info")


# handle_load_experiment, handle_switch_engine, handle_unload_model, handle_reset, handle_inject_energy
# se mantienen en pipeline_server.py por ahora ya que son muy largos y complejos
# Se pueden extraer más adelante si es necesario

HANDLERS = {
    "play": handle_play,
    "pause": handle_pause,
}

