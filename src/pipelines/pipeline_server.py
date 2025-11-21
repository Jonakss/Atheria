# src/pipeline_server.py
import asyncio
import json
import logging
import os
import uuid
from aiohttp import web
from pathlib import Path

# Asumimos la existencia y correcto funcionamiento de estos módulos locales
from .. import config as global_cfg
from ..server.server_state import g_state, broadcast, send_notification, send_to_websocket, optimize_frame_payload, get_payload_size, apply_roi_to_payload
from ..utils import get_experiment_list, load_experiment_config, get_latest_checkpoint
from ..server.server_handlers import create_experiment_handler
from .pipeline_viz import get_visualization_data
from ..model_loader import load_model
from ..engines.qca_engine import Aetheria_Motor, QuantumState
from ..analysis.analysis import analyze_universe_atlas, analyze_cell_chemistry, calculate_phase_map_metrics
from ..physics.analysis.EpochDetector import EpochDetector

# Configuración de logging - Reducir verbosidad en producción
# INFO para eventos importantes, DEBUG para detalles del bucle de simulación
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# --- WEBSOCKET HANDLER ---
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
            # Intentar obtener versión del motor C++ (requiere motor instanciado)
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

# Reemplaza esta función en tu src/pipeline_server.py
async def simulation_loop():
    """Bucle principal que evoluciona el estado y difunde los datos de visualización."""
    logging.debug("Iniciando bucle de simulación.")
    import time
    last_diagnostic_log = 0
    frame_count = 0
    # Throttle para actualizaciones de estado cuando live_feed está desactivado
    last_state_update_time = 0
    STATE_UPDATE_INTERVAL = 0.5  # Enviar actualización máximo cada 0.5 segundos (2 FPS para estado)
    
    try:
        while True:
            # CRÍTICO: Verificar is_paused al inicio de cada iteración para permitir pausa inmediata
            is_paused = g_state.get('is_paused', True)
            motor = g_state.get('motor')
            live_feed_enabled = g_state.get('live_feed_enabled', True)
            
            # Log de diagnóstico ocasional (cada 30 segundos aproximadamente - reducido para menos verbosidad)
            current_time = time.time()
            if current_time - last_diagnostic_log > 30:
                logging.debug(f"🔍 Diagnóstico: is_paused={is_paused}, motor={'✓' if motor else '✗'}, live_feed={live_feed_enabled}, step={g_state.get('simulation_step', 0)}, frames_enviados={frame_count}")
                last_diagnostic_log = current_time
            
            if motor is None and not is_paused:
                # Solo loguear ocasionalmente para no saturar
                if not hasattr(simulation_loop, '_last_warning_time'):
                    simulation_loop._last_warning_time = 0
                if current_time - simulation_loop._last_warning_time > 5:
                    logging.warning("Simulación en ejecución pero sin motor cargado. Carga un modelo para ver datos.")
                    simulation_loop._last_warning_time = current_time
            
            # Si está pausado, solo esperar y continuar (no ejecutar pasos)
            if is_paused:
                await asyncio.sleep(0.1)  # Pequeña pausa cuando está pausado para no saturar CPU
                continue
            
            if motor:
                current_step = g_state.get('simulation_step', 0)
                
                # OPTIMIZACIÓN CRÍTICA: Si live_feed está desactivado, ejecutar múltiples pasos rápidamente
                # y solo mostrar frames cada X pasos configurados (sin ralentizar la simulación)
                if not live_feed_enabled:
                    # Si live_feed está desactivado, ejecutar múltiples pasos en cada iteración
                    # para maximizar velocidad, pero solo mostrar cada X pasos
                    try:
                        # Obtener intervalo de pasos configurado (por defecto 10)
                        steps_interval = g_state.get('steps_interval', 10)
                        if 'steps_interval_counter' not in g_state:
                            g_state['steps_interval_counter'] = 0
                        if 'last_frame_sent_step' not in g_state:
                            g_state['last_frame_sent_step'] = -1  # Para forzar primer frame
                        
                        # Si steps_interval es 0 (modo manual) o -1 (modo fullspeed), ejecutar pasos pero NO enviar frames
                        # El usuario debe presionar el botón para actualizar visualización (modo manual)
                        # O nunca enviar frames (modo fullspeed)
                        if steps_interval == 0:
                            # Modo manual: ejecutar pasos rápidamente sin enviar frames
                            # Usar un valor razonable para ejecutar múltiples pasos (ej: 100)
                            steps_to_execute = 100  # Ejecutar múltiples pasos para velocidad
                        elif steps_interval == -1:
                            # Modo fullspeed: ejecutar pasos a máxima velocidad sin enviar frames
                            # Usar un valor grande para ejecutar muchos pasos en cada iteración
                            steps_to_execute = 1000  # Ejecutar muchos pasos para máxima velocidad
                        else:
                            # Ejecutar múltiples pasos en cada iteración (hasta steps_interval)
                            steps_to_execute = steps_interval
                        
                        # Medir tiempo para calcular FPS basado en pasos reales
                        steps_start_time = time.time()
                        
                        motor = g_state['motor']
                        motor_type = g_state.get('motor_type', 'unknown')
                        motor_is_native = g_state.get('motor_is_native', False)
                        
                        # Inicializar updated_step antes de usarlo (será actualizado en el bucle)
                        updated_step = current_step
                        
                        # CRÍTICO: Verificar is_paused en cada paso para permitir pausa inmediata
                        # Para motor nativo, ejecutar pasos de uno en uno para permitir pausa más frecuente
                        steps_executed_this_iteration = 0
                        for step_idx in range(steps_to_execute):
                            # Verificar si se pausó durante la ejecución (ANTES de cada paso)
                            if g_state.get('is_paused', True):
                                break  # Salir del bucle si se pausó
                            
                            # Para motor nativo: verificar pausa también ANTES de llamar a evolve_internal_state
                            # ya que el motor nativo puede ser bloqueante
                            if motor_is_native:
                                # Verificar pausa nuevamente antes de ejecutar paso nativo (más crítico)
                                if g_state.get('is_paused', True):
                                    break
                            
                            if motor:
                                motor.evolve_internal_state()
                            updated_step = current_step + step_idx + 1
                            g_state['simulation_step'] = updated_step
                            steps_executed_this_iteration += 1
                            
                            # Para motor nativo: verificar pausa también DESPUÉS de cada paso
                            # para evitar acumulación de pasos si se pausó durante la ejecución
                            if motor_is_native and g_state.get('is_paused', True):
                                break
                            
                            # Verificar qué motor se está usando (logging cada 1000 pasos, después de actualizar)
                            if updated_step % 1000 == 0 and updated_step > 0:
                                # Verificar tipo real del motor
                                actual_is_native = hasattr(motor, 'native_engine') if motor else False
                                actual_type = "native" if actual_is_native else "python"
                                if actual_type != motor_type:
                                    logging.warning(f"⚠️ Inconsistencia detectada en paso {updated_step}: motor_type en g_state={motor_type}, pero motor real={actual_type}")
                                else:
                                    logging.info(f"✅ Paso {updated_step}: Usando motor {motor_type} (confirmado)")
                        
                        # Actualizar current_step con el último valor ejecutado
                        current_step = updated_step
                        
                        steps_execution_time = time.time() - steps_start_time
                        
                        # Calcular FPS basado en pasos reales ejecutados
                        # Usar steps_executed_this_iteration en lugar de steps_to_execute
                        # porque algunos pasos pueden no haberse ejecutado si se pausó
                        actual_steps_executed = steps_executed_this_iteration if steps_executed_this_iteration > 0 else steps_to_execute
                        
                        # Evitar división por cero y valores extremos
                        if steps_execution_time > 0.0001:  # Mínimo 0.1ms para evitar valores extremos
                            steps_per_second = actual_steps_executed / steps_execution_time
                            # Limitar a un máximo razonable (ej: 10000 pasos/segundo)
                            steps_per_second = min(steps_per_second, 10000.0)
                        else:
                            steps_per_second = 0.0
                        
                        # IMPORTANTE: Distinguir entre "pasos/segundo" y "frames/segundo"
                        # Cuando live_feed está OFF, mostramos pasos/segundo
                        # Cuando live_feed está ON, mostramos frames/segundo
                        # Almacenar ambos para poder mostrar el correcto
                        g_state['steps_per_second'] = steps_per_second
                        
                        # Actualizar FPS en g_state (promediado con anterior)
                        # Para live_feed OFF: mostrar pasos/segundo (limitado a 10000)
                        # Para live_feed ON: se actualizará con frames/segundo en el bloque de visualización
                        if not live_feed_enabled:
                            # Live feed OFF: mostrar pasos/segundo
                            if 'current_fps' not in g_state or 'fps_samples' not in g_state:
                                g_state['current_fps'] = min(steps_per_second, 10000.0)
                                g_state['fps_samples'] = [min(steps_per_second, 10000.0)]
                            else:
                                # Promediar con últimos valores para suavizar
                                fps_value = min(steps_per_second, 10000.0)
                                g_state['fps_samples'].append(fps_value)
                                if len(g_state['fps_samples']) > 10:  # Mantener solo últimos 10
                                    g_state['fps_samples'].pop(0)
                                g_state['current_fps'] = sum(g_state['fps_samples']) / len(g_state['fps_samples'])
                        # Si live_feed está ON, el FPS se actualizará en el bloque de visualización
                        
                        # Actualizar contador para frames (solo si no es modo manual)
                        steps_interval_counter = g_state.get('steps_interval_counter', 0)
                        steps_interval_counter += steps_to_execute
                        g_state['steps_interval_counter'] = steps_interval_counter
                        
                        # Enviar frame cada X pasos configurados
                        # Modo manual (steps_interval = 0): NO enviar frames automáticamente
                        # También enviar frame si nunca se ha enviado uno (last_frame_sent_step == -1)
                        if steps_interval == 0:
                            # Modo manual: NO enviar frames automáticamente
                            # Solo enviar el primer frame si nunca se ha enviado uno
                            should_send_frame = (g_state['last_frame_sent_step'] == -1)
                        else:
                            # Modo automático: enviar frame cada N pasos
                            should_send_frame = (steps_interval_counter >= steps_interval) or (g_state['last_frame_sent_step'] == -1)
                        
                        if should_send_frame:
                            # Resetear contador
                            g_state['steps_interval_counter'] = 0
                            
                            # Detectar época periódicamente (cada 50 pasos para no saturar)
                            epoch_detector = g_state.get('epoch_detector')
                            if epoch_detector and updated_step % 50 == 0:
                                try:
                                    # OPTIMIZACIÓN: Para motor nativo, usar get_dense_state() si está disponible
                                    motor = g_state['motor']
                                    motor_is_native = g_state.get('motor_is_native', False)
                                    if motor_is_native and hasattr(motor, 'get_dense_state'):
                                        psi_tensor = motor.get_dense_state(check_pause_callback=lambda: g_state.get('is_paused', True))
                                    else:
                                        psi_tensor = motor.state.psi if hasattr(motor, 'state') and motor.state else None
                                    if psi_tensor is not None:
                                        # Analizar estado y determinar época
                                        metrics = epoch_detector.analyze_state(psi_tensor)
                                        epoch = epoch_detector.determine_epoch(metrics)
                                        g_state['current_epoch'] = epoch
                                        g_state['epoch_metrics'] = {
                                            'energy': float(metrics.get('energy', 0)),
                                            'clustering': float(metrics.get('clustering', 0)),
                                            'symmetry': float(metrics.get('symmetry', 0))
                                        }
                                except Exception as e:
                                    logging.debug(f"Error detectando época: {e}")
                            
                            # Calcular visualización para este frame
                            # OPTIMIZACIÓN: Solo calcular Poincaré cada N frames si está activo
                            viz_type = g_state.get('viz_type', 'density')
                            should_calc_poincare = viz_type in ['poincare', 'poincare_3d']
                            
                            # Contador para controlar frecuencia de cálculo de Poincaré
                            if should_calc_poincare:
                                if 'poincare_frame_counter' not in g_state:
                                    g_state['poincare_frame_counter'] = 0
                                g_state['poincare_frame_counter'] += 1
                                # Calcular Poincaré solo cada 3 frames cuando está activo
                                calc_poincare_this_frame = g_state['poincare_frame_counter'] % 3 == 0
                            else:
                                calc_poincare_this_frame = False
                                g_state['poincare_frame_counter'] = 0
                            
                            # OPTIMIZACIÓN CRÍTICA: Usar lazy conversion para motor nativo
                            # Solo convertir estado denso cuando se necesita visualizar
                            motor = g_state['motor']
                            motor_is_native = g_state.get('motor_is_native', False)
                            
                            # Para motor nativo: usar get_dense_state() con ROI y verificación de pausa
                            if motor_is_native and hasattr(motor, 'get_dense_state'):
                                # Obtener ROI si está habilitada
                                roi = None
                                roi_manager = g_state.get('roi_manager')
                                if roi_manager and roi_manager.roi_enabled:
                                    roi = (
                                        roi_manager.roi_x,
                                        roi_manager.roi_y,
                                        roi_manager.roi_x + roi_manager.roi_width,
                                        roi_manager.roi_y + roi_manager.roi_height
                                    )
                                
                                # Callback para verificar pausa durante conversión
                                def check_pause():
                                    return g_state.get('is_paused', True)
                                
                                # Obtener estado denso (solo convierte si es necesario)
                                psi = motor.get_dense_state(roi=roi, check_pause_callback=check_pause)
                            else:
                                # Motor Python: acceder directamente (ya es denso)
                                psi = motor.state.psi if hasattr(motor, 'state') and motor.state else None
                            
                            # Verificar que psi no sea None
                            if psi is None:
                                logging.warning("Estado psi es None. Saltando frame.")
                                continue
                            
                            delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                            viz_data = get_visualization_data(
                                psi, 
                                viz_type,
                                delta_psi=delta_psi,
                                motor=motor
                            )
                            
                            # OPTIMIZACIÓN: Reutilizar coordenadas de Poincaré del frame anterior si no se recalcula
                            if should_calc_poincare and not calc_poincare_this_frame and 'last_poincare_coords' in g_state:
                                viz_data['poincare_coords'] = g_state['last_poincare_coords']
                            elif should_calc_poincare and calc_poincare_this_frame and 'poincare_coords' in viz_data:
                                g_state['last_poincare_coords'] = viz_data['poincare_coords']
                            
                            if viz_data and isinstance(viz_data, dict):
                                map_data = viz_data.get("map_data", [])
                                if map_data and len(map_data) > 0:
                                    frame_payload_raw = {
                                        "step": updated_step,
                                        "timestamp": asyncio.get_event_loop().time(),
                                        "map_data": map_data,
                                        "hist_data": viz_data.get("hist_data", {}),
                                        "poincare_coords": viz_data.get("poincare_coords", []),
                                        "phase_attractor": viz_data.get("phase_attractor"),
                                        "flow_data": viz_data.get("flow_data"),
                                        "phase_hsv_data": viz_data.get("phase_hsv_data"),
                                        "complex_3d_data": viz_data.get("complex_3d_data"),
                                        "simulation_info": {
                                            "step": updated_step,
                                            "initial_step": g_state.get('initial_step', 0),
                                            "checkpoint_step": g_state.get('checkpoint_step', 0),
                                            "checkpoint_episode": g_state.get('checkpoint_episode', 0),
                                            "is_paused": False,
                                            "live_feed_enabled": False,
                                            "fps": g_state.get('current_fps', 0.0),
                                            "epoch": g_state.get('current_epoch', 0),
                                            "epoch_metrics": g_state.get('epoch_metrics', {})
                                        }
                                    }
                                    
                                    # Aplicar optimizaciones si están habilitadas
                                    compression_enabled = g_state.get('data_compression_enabled', True)
                                    downsample_factor = g_state.get('downsample_factor', 1)
                                    
                                    if compression_enabled or downsample_factor > 1:
                                        frame_payload = await optimize_frame_payload(
                                            frame_payload_raw,
                                            enable_compression=compression_enabled,
                                            downsample_factor=downsample_factor,
                                            viz_type=g_state.get('viz_type', 'density')
                                        )
                                    else:
                                        frame_payload = frame_payload_raw
                                    
                                    await broadcast({"type": "simulation_frame", "payload": frame_payload})
                                    frame_count += 1
                                    g_state['last_frame_sent_step'] = updated_step  # Marcar que se envió un frame
                        
                        # THROTTLE: Solo enviar actualización de estado cada STATE_UPDATE_INTERVAL segundos
                        # para evitar saturar el WebSocket con demasiados mensajes
                        current_time = time.time()
                        time_since_last_update = current_time - last_state_update_time
                        
                        if time_since_last_update >= STATE_UPDATE_INTERVAL:
                            # Enviar actualización de estado (sin datos de visualización pesados)
                            # Esto permite que el frontend muestre el progreso aunque no haya visualización
                            state_update = {
                                "step": updated_step,
                                "timestamp": asyncio.get_event_loop().time(),
                                "simulation_info": {
                                    "step": updated_step,
                                    "is_paused": False,
                                    "live_feed_enabled": False,
                                    "fps": g_state.get('current_fps', 0.0),
                                    "epoch": g_state.get('current_epoch', 0),
                                    "epoch_metrics": g_state.get('epoch_metrics', {})
                                }
                                # No incluir map_data, hist_data, etc. para ahorrar ancho de banda
                            }
                            
                            # Enviar actualización de estado (throttled para evitar saturación)
                            await broadcast({"type": "simulation_state_update", "payload": state_update})
                            last_state_update_time = current_time
                        
                        # Enviar log de simulación cada 100 pasos para no saturar los logs
                        if updated_step % 100 == 0:
                            if steps_interval == 0:
                                await broadcast({
                                    "type": "simulation_log",
                                    "payload": f"[Simulación] Paso {updated_step} completado (modo manual: presiona 'Actualizar Visualización' para ver)"
                                })
                            else:
                                await broadcast({
                                    "type": "simulation_log",
                                    "payload": f"[Simulación] Paso {updated_step} completado (live feed desactivado, mostrando cada {steps_interval} pasos)"
                                })
                            
                    except Exception as e:
                        logging.error(f"Error evolucionando estado (live_feed desactivado): {e}", exc_info=True)
                    
                    # THROTTLE ADAPTATIVO: Ajustar según live_feed y velocidad objetivo
                    # - Si live_feed está OFF: Permitir velocidades más altas sin límite rígido
                    # - Si live_feed está ON: Usar throttle mínimo para evitar CPU spin excesivo
                    simulation_speed = g_state.get('simulation_speed', 1.0)
                    target_fps = g_state.get('target_fps', 10.0)
                    base_fps = target_fps * simulation_speed
                    
                    # Calcular sleep time ideal
                    ideal_sleep = 1.0 / base_fps if base_fps > 0 else 0.001
                    
                    # THROTTLE ADAPTATIVO:
                    # - Live feed OFF: Permitir velocidades muy altas (mínimo yield para cooperar con event loop)
                    # - Live feed ON: Usar throttle mínimo para evitar CPU spin excesivo
                    if not live_feed_enabled:
                        # Sin live feed: Permitir velocidades más altas, pero yield para cooperar con event loop
                        # Solo usar sleep si el ideal sleep es > 1ms (velocidades razonables)
                        if ideal_sleep > 0.001:
                            await asyncio.sleep(ideal_sleep)
                        else:
                            # Velocidad muy alta: yield para permitir otros tasks, pero sin sleep
                            await asyncio.sleep(0)  # Yield al event loop
                    else:
                        # Con live feed: Usar throttle mínimo para evitar CPU spin
                        sleep_time = max(0.016, ideal_sleep)  # Mínimo 16ms cuando hay live feed
                        await asyncio.sleep(sleep_time)
                    continue
                
                try:
                    # Evolucionar el estado solo si live_feed está activo
                    g_state['motor'].evolve_internal_state()
                    g_state['simulation_step'] = current_step + 1
                    
                    # Validar que el motor tenga un estado válido
                    if g_state['motor'].state.psi is None:
                        logging.warning("Motor activo pero sin estado psi. Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Detectar época periódicamente (cada 50 pasos para no saturar)
                    epoch_detector = g_state.get('epoch_detector')
                    if epoch_detector:
                        current_step = g_state.get('simulation_step', 0)
                        if current_step % 50 == 0:
                            try:
                                psi_tensor = g_state['motor'].state.psi
                                if psi_tensor is not None:
                                    # Analizar estado y determinar época
                                    metrics = epoch_detector.analyze_state(psi_tensor)
                                    epoch = epoch_detector.determine_epoch(metrics)
                                    g_state['current_epoch'] = epoch
                                    g_state['epoch_metrics'] = {
                                        'energy': float(metrics.get('energy', 0)),
                                        'clustering': float(metrics.get('clustering', 0)),
                                        'symmetry': float(metrics.get('symmetry', 0))
                                    }
                            except Exception as e:
                                logging.debug(f"Error detectando época: {e}")
                    
                    # --- CALCULAR VISUALIZACIONES SOLO SI LIVE_FEED ESTÁ ACTIVO ---
                    # OPTIMIZACIÓN CRÍTICA: Usar lazy conversion para motor nativo
                    # Solo convertir estado denso cuando se necesita visualizar
                    motor = g_state['motor']
                    motor_is_native = g_state.get('motor_is_native', False)
                    
                    # Para motor nativo: usar get_dense_state() con ROI y verificación de pausa
                    if motor_is_native and hasattr(motor, 'get_dense_state'):
                        # Obtener ROI si está habilitada
                        roi = None
                        roi_manager = g_state.get('roi_manager')
                        if roi_manager and roi_manager.roi_enabled:
                            roi = (
                                roi_manager.roi_x,
                                roi_manager.roi_y,
                                roi_manager.roi_x + roi_manager.roi_width,
                                roi_manager.roi_y + roi_manager.roi_height
                            )
                        
                        # Callback para verificar pausa durante conversión
                        def check_pause():
                            return g_state.get('is_paused', True)
                        
                        # Obtener estado denso (solo convierte si es necesario)
                        psi = motor.get_dense_state(roi=roi, check_pause_callback=check_pause)
                    else:
                        # Motor Python: acceder directamente (ya es denso)
                        psi = motor.state.psi if hasattr(motor, 'state') and motor.state else None
                    
                    # Verificar que psi no sea None
                    if psi is None:
                        logging.warning("Estado psi es None. Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Optimización: Usar inference_mode para mejor rendimiento GPU
                    # Obtener delta_psi si está disponible para visualizaciones de flujo
                    delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                    viz_data = get_visualization_data(
                        psi, 
                        g_state.get('viz_type', 'density'),
                        delta_psi=delta_psi,
                        motor=motor
                    )
                    
                    # Validar que viz_data tenga los campos necesarios
                    if not viz_data or not isinstance(viz_data, dict):
                        logging.warning(f"⚠️ get_visualization_data retornó datos inválidos (tipo: {type(viz_data)}). Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Validar que map_data no esté vacío
                    map_data = viz_data.get("map_data", [])
                    if not map_data or (isinstance(map_data, list) and len(map_data) == 0):
                        logging.warning(f"⚠️ map_data está vacío en step {g_state.get('simulation_step', 0)}. Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Construir frame_payload con información completa del paso del tiempo
                    # IMPORTANTE: Usar el step actualizado después de evolve_internal_state
                    updated_step = g_state.get('simulation_step', 0)
                    current_time = asyncio.get_event_loop().time()
                    frame_payload_raw = {
                        "step": updated_step,  # Usar el step actualizado
                        "timestamp": current_time,
                        "map_data": map_data,  # Ya validado arriba
                        "hist_data": viz_data.get("hist_data", {}),
                        "poincare_coords": viz_data.get("poincare_coords", []),
                        "phase_attractor": viz_data.get("phase_attractor"),
                        "flow_data": viz_data.get("flow_data"),
                        "phase_hsv_data": viz_data.get("phase_hsv_data"),
                        "complex_3d_data": viz_data.get("complex_3d_data"),
                        # Información adicional para la UI
                        "simulation_info": {
                            "step": updated_step,
                            "initial_step": g_state.get('initial_step', 0),
                            "checkpoint_step": g_state.get('checkpoint_step', 0),
                            "checkpoint_episode": g_state.get('checkpoint_episode', 0),
                            "is_paused": False,
                            "live_feed_enabled": live_feed_enabled,
                            "fps": g_state.get('current_fps', 0.0),
                            "epoch": g_state.get('current_epoch', 0),
                            "epoch_metrics": g_state.get('epoch_metrics', {}),
                            "training_grid_size": g_state.get('training_grid_size'),
                            "inference_grid_size": g_state.get('inference_grid_size'),
                            "grid_scaled": g_state.get('grid_size_ratio', 1.0) != 1.0
                        }
                    }
                    
                    # Optimizar payload (ROI, compresión y downsampling)
                    # 1. Aplicar ROI primero (reduce el tamaño de los datos)
                    roi_manager = g_state.get('roi_manager')
                    if roi_manager and roi_manager.roi_enabled:
                        from ..managers.roi_manager import apply_roi_to_payload
                        frame_payload_roi = apply_roi_to_payload(frame_payload_raw, roi_manager)
                    else:
                        frame_payload_roi = frame_payload_raw
                    
                    # 2. Aplicar compresión y downsampling
                    compression_enabled = g_state.get('data_compression_enabled', True)
                    downsample_factor = g_state.get('downsample_factor', 1)
                    viz_type = g_state.get('viz_type', 'density')
                    
                    # Por ahora, solo aplicar optimización si está habilitada explícitamente
                    # y el payload es grande (para no afectar rendimiento con payloads pequeños)
                    if compression_enabled or downsample_factor > 1:
                        frame_payload = await optimize_frame_payload(
                            frame_payload_roi,
                            enable_compression=compression_enabled,
                            downsample_factor=downsample_factor,
                            viz_type=viz_type
                        )
                        
                        # Logging ocasional del tamaño del payload (cada 100 frames)
                        if updated_step % 100 == 0:
                            original_size = get_payload_size(frame_payload_raw)
                            optimized_size = get_payload_size(frame_payload)
                            compression_ratio = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0
                            roi_info = frame_payload.get('roi_info', {})
                            roi_msg = f" (ROI: {roi_info.get('reduction_ratio', 1.0):.1f}x reducción)" if roi_info.get('enabled') else ""
                            logging.debug(f"Payload size: {original_size/1024:.1f}KB → {optimized_size/1024:.1f}KB ({compression_ratio:.1f}% reducción){roi_msg}")
                    else:
                        frame_payload = frame_payload_roi
                    
                    # Guardar en historial si está habilitado
                    # IMPORTANTE: Solo guardar si live_feed está activo para evitar guardar frames vacíos
                    if g_state.get('history_enabled', False) and live_feed_enabled:
                        try:
                            # Solo guardar cada N frames para reducir uso de memoria
                            # Por defecto, guardar cada 10 frames (reducción de 10x en memoria)
                            history_interval = g_state.get('history_save_interval', 10)
                            if updated_step % history_interval == 0:
                                g_state['simulation_history'].add_frame(frame_payload)
                        except Exception as e:
                            logging.debug(f"Error guardando frame en historial: {e}")
                    
                    # Enviar frame solo si live_feed está activo
                    # Verificar que el payload tenga step antes de enviar
                    if 'step' not in frame_payload:
                        logging.warning(f"⚠️ Frame sin step, añadiendo step={updated_step}")
                        frame_payload['step'] = updated_step
                    
                    await broadcast({"type": "simulation_frame", "payload": frame_payload})
                    frame_count += 1
                    
                    # CRÍTICO: Calcular FPS de frames cuando live_feed está ON
                    # Cuando live_feed está ON, mostrar frames/segundo (no pasos/segundo)
                    if live_feed_enabled:
                        # Calcular FPS basado en frames reales enviados
                        if 'last_frame_sent_time' not in g_state:
                            g_state['last_frame_sent_time'] = time.time()
                            g_state['frame_fps_samples'] = []
                        
                        current_time = time.time()
                        last_frame_time = g_state.get('last_frame_sent_time', current_time)
                        delta_time = current_time - last_frame_time
                        
                        if delta_time > 0:
                            # Calcular FPS instantáneo (1 frame / delta_time)
                            instant_fps = 1.0 / delta_time
                            
                            # Promediar para suavizar
                            if 'frame_fps_samples' not in g_state:
                                g_state['frame_fps_samples'] = []
                            g_state['frame_fps_samples'].append(instant_fps)
                            
                            # Mantener solo últimos 30 samples (aproximadamente 0.5-1 segundo a 30-60 FPS)
                            if len(g_state['frame_fps_samples']) > 30:
                                g_state['frame_fps_samples'].pop(0)
                            
                            # Calcular promedio
                            if len(g_state['frame_fps_samples']) > 0:
                                avg_frame_fps = sum(g_state['frame_fps_samples']) / len(g_state['frame_fps_samples'])
                                # Limitar a máximo razonable (ej: 120 FPS para frames)
                                g_state['current_fps'] = min(avg_frame_fps, 120.0)
                        
                        g_state['last_frame_sent_time'] = current_time
                    
                    # Logging ocasional para debug (cada 100 frames para reducir overhead)
                    if updated_step % 100 == 0:
                        logging.debug(f"✅ Frame {updated_step} enviado. FPS: {g_state.get('current_fps', 0):.1f}")
                    
                    # OPTIMIZACIÓN: Enviar log de simulación con menor frecuencia (cada 100 pasos)
                    # Reducir overhead de WebSocket
                    if updated_step % 100 == 0:
                        await broadcast({
                            "type": "simulation_log",
                            "payload": f"[Simulación] Paso {updated_step} completado"
                        })
                    
                    # Capturar snapshot para análisis t-SNE (cada N pasos) - OPTIMIZADO
                    # Solo capturar si está habilitado y en el intervalo correcto
                    snapshot_interval = g_state.get('snapshot_interval', 500)  # Por defecto cada 500 pasos (más espaciado)
                    snapshot_enabled = g_state.get('snapshot_enabled', False)  # Deshabilitado por defecto para no afectar rendimiento
                    
                    if snapshot_enabled and updated_step % snapshot_interval == 0:
                        if 'snapshots' not in g_state:
                            g_state['snapshots'] = []
                        
                        # Optimización: usar detach() antes de clone() para evitar grafo computacional
                        # y mover a CPU de forma asíncrona si es necesario
                        try:
                            psi_tensor = g_state['motor'].state.psi
                            # Detach y clonar de forma más eficiente, mover a CPU inmediatamente
                            with torch.no_grad():
                                snapshot = psi_tensor.detach().cpu().clone() if hasattr(psi_tensor, 'detach') else psi_tensor.cpu().clone()
                            
                            g_state['snapshots'].append({
                                'psi': snapshot,
                                'step': updated_step,
                                'timestamp': asyncio.get_event_loop().time()
                            })
                            
                            # Limitar número de snapshots almacenados (mantener últimos 500 para reducir memoria)
                            max_snapshots = g_state.get('max_snapshots', 500)
                            if len(g_state['snapshots']) > max_snapshots:
                                # Liberar memoria de los snapshots más antiguos antes de eliminarlos
                                old_snapshots = g_state['snapshots'][:-max_snapshots]
                                for old_snap in old_snapshots:
                                    if 'psi' in old_snap and old_snap['psi'] is not None:
                                        del old_snap['psi']  # Liberar tensor explícitamente
                                
                                # Eliminar los más antiguos de forma eficiente
                                g_state['snapshots'] = g_state['snapshots'][-max_snapshots:]
                                
                                # Forzar garbage collection para liberar memoria inmediatamente
                                import gc
                                gc.collect()
                        except Exception as e:
                            # Si falla la captura, no afectar la simulación
                            logging.debug(f"Error capturando snapshot en paso {updated_step}: {e}")
                    
                except Exception as e:
                    logging.error(f"Error en el bucle de simulación: {e}", exc_info=True)
                    # Continuar el bucle en lugar de detenerlo
                    await asyncio.sleep(0.1)
                    continue
            
            # Controla la velocidad de la simulación según simulation_speed y target_fps
            simulation_speed = g_state.get('simulation_speed', 1.0)
            target_fps = g_state.get('target_fps', 10.0)
            frame_skip = g_state.get('frame_skip', 0)
            
            # THROTTLE ADAPTATIVO: Ajustar según estado y velocidad objetivo
            # - Live feed OFF: Permitir velocidades más altas sin límite rígido
            # - Live feed ON: Usar throttle mínimo para evitar CPU spin excesivo
            base_fps = target_fps * simulation_speed
            ideal_sleep = 1.0 / base_fps if base_fps > 0 else 0.001
            
            # Aplicar throttle adaptativo según live_feed
            live_feed_enabled = g_state.get('live_feed_enabled', True)
            if not live_feed_enabled and ideal_sleep < 0.001:
                # Sin live feed + velocidad muy alta: yield sin sleep (cooperar con event loop)
                sleep_time = 0
            elif not live_feed_enabled:
                # Sin live feed + velocidad razonable: usar sleep calculado (sin mínimo)
                sleep_time = ideal_sleep
            else:
                # Con live feed: usar throttle mínimo para evitar CPU spin excesivo
                sleep_time = max(0.016, ideal_sleep)  # Mínimo 16ms cuando hay live feed
            
            # Aplicar frame skip solo si live_feed está OFF
            # Cuando live_feed está ON, siempre enviamos frames (no saltamos)
            live_feed_enabled = g_state.get('live_feed_enabled', True)
            if frame_skip > 0 and not live_feed_enabled and g_state.get('simulation_step', 0) % (frame_skip + 1) != 0:
                # Saltar frame: solo evolución, no visualización
                # SOLO cuando live_feed está OFF
                if not is_paused and motor:
                    try:
                        motor = g_state['motor']
                        if motor:
                            motor.evolve_internal_state()
                        g_state['simulation_step'] = g_state.get('simulation_step', 0) + 1
                    except:
                        pass
            
            # Sleep adaptativo: usar yield si sleep_time es 0, sino usar sleep normal
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # Yield al event loop para permitir otros tasks (sin delay)
                await asyncio.sleep(0)
    except (SystemExit, asyncio.CancelledError):
        # Shutdown graceful - no loguear como error
        logging.info("Bucle de simulación detenido (shutdown graceful)")
        raise
    except Exception as e:
        logging.error(f"Error crítico en el bucle de simulación: {e}", exc_info=True)
        await broadcast({
            "type": "simulation_log",
            "payload": f"[Error] Error en simulación: {str(e)}"
        })
        g_state['is_paused'] = True
        await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
        await asyncio.sleep(2)

# --- Definición de Handlers para los Comandos ---

async def handle_create_experiment(args):
    args['CONTINUE_TRAINING'] = False
    asyncio.create_task(create_experiment_handler(args))

async def handle_continue_experiment(args):
    """
    Continúa el entrenamiento de un experimento existente.
    Carga la configuración guardada y la combina con los argumentos del frontend.
    """
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    
    if not exp_name:
        if ws: await send_notification(ws, "❌ El nombre del experimento es obligatorio.", "error")
        return
    
    try:
        # Cargar la configuración del experimento guardada
        config = load_experiment_config(exp_name)
        if not config:
            msg = f"❌ No se encontró la configuración para '{exp_name}'. Asegúrate de que el experimento existe."
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        # Construir los argumentos completos combinando la config guardada con los del frontend
        # Usar getattr con valores por defecto para evitar errores si faltan campos
        continue_args = {
            'ws_id': args.get('ws_id'),
            'EXPERIMENT_NAME': exp_name,
            'MODEL_ARCHITECTURE': getattr(config, 'MODEL_ARCHITECTURE', None),
            'LR_RATE_M': getattr(config, 'LR_RATE_M', None),
            'GRID_SIZE_TRAINING': getattr(config, 'GRID_SIZE_TRAINING', None),
            'QCA_STEPS_TRAINING': getattr(config, 'QCA_STEPS_TRAINING', None),
            'CONTINUE_TRAINING': True,
        }
        
        # Validar que todos los campos requeridos estén presentes
        required_fields = ['MODEL_ARCHITECTURE', 'LR_RATE_M', 'GRID_SIZE_TRAINING', 'QCA_STEPS_TRAINING']
        missing_fields = [field for field in required_fields if continue_args[field] is None]
        if missing_fields:
            msg = f"❌ La configuración del experimento '{exp_name}' está incompleta. Faltan: {', '.join(missing_fields)}"
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        # Manejar EPISODES_TO_ADD: puede ser episodios adicionales o el total nuevo
        episodes_to_add = args.get('EPISODES_TO_ADD')
        if episodes_to_add:
            # Si hay un checkpoint, sumar los episodios adicionales al total actual
            checkpoint_path = get_latest_checkpoint(exp_name)
            if checkpoint_path:
                # Intentar extraer el episodio actual del checkpoint
                try:
                    import torch
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    current_episode = checkpoint.get('episode', config.TOTAL_EPISODES)
                    continue_args['TOTAL_EPISODES'] = current_episode + episodes_to_add
                    logging.info(f"Continuando desde episodio {current_episode}, añadiendo {episodes_to_add} más. Total: {continue_args['TOTAL_EPISODES']}")
                except Exception as e:
                    logging.warning(f"No se pudo leer el episodio del checkpoint, usando TOTAL_EPISODES de la config: {e}")
                    continue_args['TOTAL_EPISODES'] = config.TOTAL_EPISODES + episodes_to_add
            else:
                # Sin checkpoint, usar el valor de la config + episodios adicionales
                continue_args['TOTAL_EPISODES'] = config.TOTAL_EPISODES + episodes_to_add
        else:
            # Si no se proporciona EPISODES_TO_ADD, usar el TOTAL_EPISODES de la config
            continue_args['TOTAL_EPISODES'] = config.TOTAL_EPISODES
        
        # Convertir MODEL_PARAMS de SimpleNamespace a dict para JSON
        # Usar la función recursiva de utils para manejar casos anidados
        from ..utils import sns_to_dict_recursive
        if hasattr(config, 'MODEL_PARAMS') and config.MODEL_PARAMS is not None:
            model_params = config.MODEL_PARAMS
            continue_args['MODEL_PARAMS'] = sns_to_dict_recursive(model_params)
            # Validar que MODEL_PARAMS no esté vacío
            if not continue_args['MODEL_PARAMS'] or (isinstance(continue_args['MODEL_PARAMS'], dict) and len(continue_args['MODEL_PARAMS']) == 0):
                msg = f"❌ MODEL_PARAMS está vacío en la configuración de '{exp_name}'. No se puede continuar el entrenamiento."
                logging.error(msg)
                if ws: await send_notification(ws, msg, "error")
                return
        else:
            # Fallback si no hay MODEL_PARAMS en la config
            msg = f"❌ No se encontró MODEL_PARAMS en la configuración de '{exp_name}'. No se puede continuar el entrenamiento."
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        # Validar que MODEL_PARAMS esté presente antes de continuar
        if 'MODEL_PARAMS' not in continue_args or continue_args['MODEL_PARAMS'] is None:
            msg = f"❌ MODEL_PARAMS es requerido para continuar el entrenamiento de '{exp_name}'."
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        # Llamar al handler de creación con los argumentos completos
        asyncio.create_task(create_experiment_handler(continue_args))
        
    except Exception as e:
        logging.error(f"Error al continuar el entrenamiento de '{exp_name}': {e}", exc_info=True)
        if ws: await send_notification(ws, f"❌ Error al continuar el entrenamiento: {str(e)}", "error")

async def handle_stop_training(args):
    ws = g_state['websockets'].get(args['ws_id'])
    logging.info(f"Recibida orden de detener entrenamiento de [{args['ws_id']}]")
    if g_state.get('training_process'):
        try:
            g_state['training_process'].kill()
            await g_state['training_process'].wait()
        except ProcessLookupError:
            logging.warning("El proceso de entrenamiento ya había terminado.")
        finally:
            g_state['training_process'] = None
            await broadcast({"type": "training_status_update", "payload": {"status": "idle"}})
            if ws: await send_notification(ws, "Entrenamiento detenido por el usuario.", "info")

async def handle_update_visualization(args):
    """Actualiza la visualización manualmente (útil cuando steps_interval = 0)."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    motor = g_state.get('motor')
    
    if not motor:
        msg = "⚠️ No hay modelo cargado. Carga un experimento primero."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    if not motor.state or motor.state.psi is None:
        msg = "⚠️ El modelo cargado no tiene un estado válido."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    try:
        # Obtener estado actual
        motor_is_native = g_state.get('motor_is_native', False)
        if motor_is_native and hasattr(motor, 'get_dense_state'):
            # Motor nativo: usar lazy conversion
            roi = None
            roi_manager = g_state.get('roi_manager')
            if roi_manager and roi_manager.roi_enabled:
                roi = (
                    roi_manager.roi_x,
                    roi_manager.roi_y,
                    roi_manager.roi_x + roi_manager.roi_width,
                    roi_manager.roi_y + roi_manager.roi_height
                )
            psi = motor.get_dense_state(roi=roi, check_pause_callback=lambda: g_state.get('is_paused', True))
        else:
            # Motor Python: acceder directamente
            psi = motor.state.psi if hasattr(motor, 'state') and motor.state else None
        
        if psi is None:
            msg = "⚠️ No se pudo obtener el estado actual."
            logging.warning(msg)
            if ws: await send_notification(ws, msg, "warning")
            return
        
        # Generar datos de visualización
        viz_type = g_state.get('viz_type', 'density')
        delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
        viz_data = get_visualization_data(psi, viz_type, delta_psi=delta_psi, motor=motor)
        
        if not viz_data or not isinstance(viz_data, dict):
            msg = "⚠️ Error generando datos de visualización."
            logging.warning(msg)
            if ws: await send_notification(ws, msg, "warning")
            return
        
        current_step = g_state.get('simulation_step', 0)
        live_feed_enabled = g_state.get('live_feed_enabled', False)
        
        frame_payload_raw = {
            "step": current_step,
            "timestamp": asyncio.get_event_loop().time(),
            "map_data": viz_data.get("map_data", []),
            "hist_data": viz_data.get("hist_data", {}),
            "poincare_coords": viz_data.get("poincare_coords", []),
            "phase_attractor": viz_data.get("phase_attractor"),
            "flow_data": viz_data.get("flow_data"),
            "phase_hsv_data": viz_data.get("phase_hsv_data"),
            "complex_3d_data": viz_data.get("complex_3d_data"),
            "simulation_info": {
                "step": current_step,
                "initial_step": g_state.get('initial_step', 0),
                "checkpoint_step": g_state.get('checkpoint_step', 0),
                "checkpoint_episode": g_state.get('checkpoint_episode', 0),
                "is_paused": g_state.get('is_paused', True),
                "live_feed_enabled": live_feed_enabled,
                "fps": g_state.get('current_fps', 0.0),
                "epoch": g_state.get('current_epoch', 0),
                "epoch_metrics": g_state.get('epoch_metrics', {}),
                "training_grid_size": g_state.get('training_grid_size'),
                "inference_grid_size": g_state.get('inference_grid_size'),
                "grid_scaled": g_state.get('grid_size_ratio', 1.0) != 1.0
            }
        }
        
        # Aplicar optimizaciones si están habilitadas
        compression_enabled = g_state.get('data_compression_enabled', True)
        downsample_factor = g_state.get('downsample_factor', 1)
        
        if compression_enabled or downsample_factor > 1:
            frame_payload = await optimize_frame_payload(
                frame_payload_raw,
                enable_compression=compression_enabled,
                downsample_factor=downsample_factor,
                viz_type=viz_type
            )
        else:
            frame_payload = frame_payload_raw
        
        # Enviar frame a todos los clientes conectados
        await broadcast({"type": "simulation_frame", "payload": frame_payload})
        g_state['last_frame_sent_step'] = current_step  # Actualizar último frame enviado
        
        msg = f"✅ Visualización actualizada (paso {current_step})"
        logging.info(msg)
        if ws: await send_notification(ws, msg, "success")
        
    except Exception as e:
        logging.error(f"Error actualizando visualización: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al actualizar visualización: {str(e)}", "error")

async def handle_set_viz(args):
    viz_type = args.get("viz_type", "density")
    g_state['viz_type'] = viz_type
    if (ws := g_state['websockets'].get(args.get('ws_id'))):
        await send_notification(ws, f"Visualización cambiada a: {viz_type}", "info")
    # Si hay un motor activo, enviar un frame actualizado inmediatamente
    # SOLO si live_feed está habilitado
    live_feed_enabled = g_state.get('live_feed_enabled', True)
    if g_state.get('motor') and live_feed_enabled:
        try:
            motor = g_state['motor']
            # OPTIMIZACIÓN CRÍTICA: Usar lazy conversion para motor nativo
            motor_is_native = hasattr(motor, 'native_engine')
            if motor_is_native and hasattr(motor, 'get_dense_state'):
                # Para motor nativo, usar get_dense_state() solo cuando se necesita visualizar
                psi = motor.get_dense_state(check_pause_callback=lambda: g_state.get('is_paused', True))
            else:
                # Motor Python: acceder directamente (ya es denso)
                psi = motor.state.psi if hasattr(motor, 'state') and motor.state else None
            
            if psi is None:
                logging.warning("Motor activo pero sin estado psi. No se puede actualizar visualización.")
                return
            
            delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
            viz_data = get_visualization_data(psi, viz_type, delta_psi=delta_psi, motor=motor)
            if not viz_data or not isinstance(viz_data, dict):
                logging.warning("get_visualization_data retornó datos inválidos.")
                return
            
            current_step = g_state.get('simulation_step', 0)
            frame_payload_raw = {
                "step": current_step,
                "timestamp": asyncio.get_event_loop().time(),
                "map_data": viz_data.get("map_data", []),
                "hist_data": viz_data.get("hist_data", {}),
                "poincare_coords": viz_data.get("poincare_coords", []),
                "phase_attractor": viz_data.get("phase_attractor"),
                "flow_data": viz_data.get("flow_data"),
                "phase_hsv_data": viz_data.get("phase_hsv_data"),
                "complex_3d_data": viz_data.get("complex_3d_data"),
                "simulation_info": {
                    "step": current_step,
                    "is_paused": g_state.get('is_paused', True),
                    "live_feed_enabled": live_feed_enabled,
                    "fps": g_state.get('current_fps', 0.0)
                }
            }
            
            # Aplicar optimizaciones si están habilitadas
            compression_enabled = g_state.get('data_compression_enabled', True)
            downsample_factor = g_state.get('downsample_factor', 1)
            
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

async def handle_play(args):
    ws = g_state['websockets'].get(args['ws_id'])
    
    # Validar que haya un motor cargado antes de iniciar
    motor = g_state.get('motor')
    if not motor:
        msg = "⚠️ No hay un modelo cargado. Primero debes cargar un experimento entrenado."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    # Validar que el motor tenga estado válido
    if not motor.state or motor.state.psi is None:
        msg = "⚠️ El modelo cargado no tiene un estado válido. Intenta reiniciar la simulación."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    g_state['is_paused'] = False
    logging.info(f"Simulación iniciada. Motor: {type(motor).__name__}, Step: {g_state.get('simulation_step', 0)}, Live feed: {g_state.get('live_feed_enabled', True)}")
    await broadcast({"type": "inference_status_update", "payload": {"status": "running"}})
    if ws: await send_notification(ws, "Simulación iniciada.", "info")

async def handle_pause(args):
    ws = g_state['websockets'].get(args.get('ws_id'))
    logging.info("Comando de pausa recibido. Pausando simulación...")
    g_state['is_paused'] = True
    await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
    if ws:
        await send_notification(ws, "Simulación pausada.", "info")

async def handle_load_experiment(args):
    ws = g_state['websockets'].get(args['ws_id'])
    exp_name = args.get("experiment_name")
    if not exp_name:
        if ws: await send_notification(ws, "Nombre de experimento no proporcionado.", "error")
        return

    # Inicializar device_str al inicio para evitar UnboundLocalError
    # SIEMPRE intentar usar el mejor dispositivo disponible (CUDA primero si está disponible)
    import torch
    from .. import config as global_cfg
    # Usar get_device() que ya tiene lógica robusta de detección
    device = global_cfg.DEVICE
    device_str = str(device).split(':')[0]  # 'cuda' o 'cpu'

    try:
        logging.info(f"Intentando cargar el experimento '{exp_name}' para [{args['ws_id']}]...")
        if ws: await send_notification(ws, f"Cargando modelo '{exp_name}'...", "info")
        
        # Inicializar EpochDetector si no existe
        if 'epoch_detector' not in g_state:
            g_state['epoch_detector'] = EpochDetector()
        
        # OPTIMIZACIÓN DE MEMORIA: Liberar motor anterior antes de cargar uno nuevo
        old_motor = g_state.get('motor')
        if old_motor is not None:
            try:
                # Limpiar snapshots y historial del motor anterior
                if 'snapshots' in g_state:
                    for snap in g_state['snapshots']:
                        if isinstance(snap, dict) and 'psi' in snap and snap['psi'] is not None:
                            if hasattr(snap['psi'], 'detach'):
                                del snap['psi']
                    g_state['snapshots'] = []
                
                # Limpiar historial si está habilitado
                if 'simulation_history' in g_state:
                    g_state['simulation_history'].clear()
                
                # CRÍTICO: Limpiar motor nativo explícitamente antes de eliminarlo
                # Esto previene segfaults al destruir el motor nativo C++
                if hasattr(old_motor, 'native_engine'):
                    # Es un motor nativo - llamar cleanup explícitamente
                    try:
                        if hasattr(old_motor, 'cleanup'):
                            old_motor.cleanup()
                            logging.debug("Motor nativo limpiado explícitamente antes de eliminarlo")
                        else:
                            # Fallback: limpiar manualmente si no hay método cleanup
                            if hasattr(old_motor, 'native_engine') and old_motor.native_engine is not None:
                                old_motor.native_engine = None
                            if hasattr(old_motor, 'state') and old_motor.state is not None:
                                if hasattr(old_motor.state, 'psi') and old_motor.state.psi is not None:
                                    old_motor.state.psi = None
                                old_motor.state = None
                    except Exception as cleanup_error:
                        logging.warning(f"Error durante cleanup de motor nativo: {cleanup_error}")
                
                # Remover referencia del estado global antes de destruir
                g_state['motor'] = None
                
                # Liberar motor anterior
                del old_motor
                import gc
                gc.collect()  # Forzar garbage collection
                logging.debug("Motor anterior liberado para liberar memoria")
            except Exception as e:
                logging.warning(f"Error liberando motor anterior: {e}", exc_info=True)
        
        config = load_experiment_config(exp_name)
        if not config:
            msg = f"❌ No se encontró la configuración para '{exp_name}'. Asegúrate de que el experimento existe."
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        # Asegurar que GAMMA_DECAY esté presente en la configuración (para término Lindbladian)
        # config es un SimpleNamespace, usar hasattr() en lugar de 'in'
        if not hasattr(config, 'GAMMA_DECAY') or getattr(config, 'GAMMA_DECAY', None) is None:
            config.GAMMA_DECAY = getattr(global_cfg, 'GAMMA_DECAY', 0.01)
            logging.info(f"GAMMA_DECAY no encontrado en config, usando valor por defecto: {config.GAMMA_DECAY}")
        
        # Asegurar que INITIAL_STATE_MODE_INFERENCE esté presente en la configuración
        if not hasattr(config, 'INITIAL_STATE_MODE_INFERENCE') or getattr(config, 'INITIAL_STATE_MODE_INFERENCE', None) is None:
            config.INITIAL_STATE_MODE_INFERENCE = getattr(global_cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
            logging.info(f"INITIAL_STATE_MODE_INFERENCE no encontrado en config, usando valor por defecto: {config.INITIAL_STATE_MODE_INFERENCE}")
        
        # CRÍTICO: Siempre usar el último checkpoint disponible
        checkpoint_path = get_latest_checkpoint(exp_name)
        model = None
        state_dict = None
        checkpoint_step = 0  # Paso guardado en el checkpoint
        checkpoint_episode = 0  # Episodio guardado en el checkpoint
        
        if checkpoint_path:
            # Cargar información del checkpoint ANTES de cargar el modelo
            try:
                import torch
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                
                # Obtener step/episode del checkpoint si está disponible
                if isinstance(checkpoint_data, dict):
                    checkpoint_step = checkpoint_data.get('step', checkpoint_data.get('simulation_step', 0))
                    checkpoint_episode = checkpoint_data.get('episode', 0)
                    
                    # Si no hay 'step', intentar calcular desde episode y steps_per_episode
                    if checkpoint_step == 0 and checkpoint_episode > 0:
                        steps_per_episode = getattr(config, 'QCA_STEPS_TRAINING', getattr(config, 'STEPS_PER_EPISODE', 100))
                        checkpoint_step = checkpoint_episode * steps_per_episode
                        logging.info(f"⚠️ Checkpoint no tiene 'step', calculado desde episode: {checkpoint_episode} × {steps_per_episode} = {checkpoint_step}")
                    
                    if checkpoint_step > 0:
                        logging.info(f"📊 Checkpoint encontrado: episode={checkpoint_episode}, step={checkpoint_step}")
                        if ws: await send_notification(ws, f"📊 Checkpoint: episodio {checkpoint_episode}, paso {checkpoint_step}", "info")
                else:
                    logging.warning(f"⚠️ Checkpoint tiene formato inesperado, no se puede leer step/episode")
            except Exception as e:
                logging.warning(f"⚠️ No se pudo leer información del checkpoint: {e}")
            
            # Cargar modelo desde checkpoint (modelo entrenado)
            logging.info(f"📋 Paso 4/10: Cargando modelo desde checkpoint...")
            model, state_dict = load_model(config, checkpoint_path)
            if model is None:
                msg = f"❌ Error al cargar el modelo desde el checkpoint. Verifica que el checkpoint no esté corrupto."
                logging.error(msg)
                if ws: await send_notification(ws, msg, "error")
                return
            logging.info(f"✅ Paso 4/10: Modelo cargado desde checkpoint: {checkpoint_path}")
        else:
            # No hay checkpoint: crear modelo nuevo sin pesos entrenados
            logging.info(f"⚠️ El experimento '{exp_name}' no tiene checkpoints. Creando modelo nuevo sin pesos entrenados.")
            if ws: await send_notification(ws, f"⚠️ Sin checkpoint. Iniciando con modelo nuevo (ruido aleatorio).", "info")
            
            try:
                from ..model_loader import create_new_model
                model = create_new_model(config)
                if model is None:
                    msg = f"❌ Error al crear el modelo desde la configuración."
                    logging.error(msg)
                    if ws: await send_notification(ws, msg, "error")
                    return
                logging.info(f"✅ Modelo nuevo creado desde configuración (sin pesos entrenados)")
            except Exception as e:
                msg = f"❌ Error al crear modelo nuevo: {str(e)}"
                logging.error(msg, exc_info=True)
                if ws: await send_notification(ws, msg, "error")
                return
        
        # Asegurar que el modelo esté en modo evaluación para inferencia
        model.eval()
        
        d_state = config.MODEL_PARAMS.d_state
        
        # Obtener tamaños de grid
        training_grid_size = getattr(config, 'GRID_SIZE_TRAINING', global_cfg.GRID_SIZE_TRAINING)
        inference_grid_size = global_cfg.GRID_SIZE_INFERENCE
        
        # Nota: Los modelos convolucionales (UNet, MLP, etc.) pueden manejar diferentes tamaños de grid
        # El entrenamiento puede hacerse en un grid pequeño (ej: 64x64) y la inferencia en uno grande (ej: 256x256)
        if training_grid_size and training_grid_size != inference_grid_size:
            logging.info(f"Escalando de grid de entrenamiento ({training_grid_size}x{training_grid_size}) a grid de inferencia ({inference_grid_size}x{inference_grid_size})")
        
        # --- INTEGRACIÓN DEL MOTOR NATIVO (C++) ---
        # Intentar usar el motor nativo de alto rendimiento si está disponible
        # El motor nativo es 250-400x más rápido que el motor Python
        # NOTA: Solo usar motor nativo si hay checkpoint (modelo entrenado)
        # Si no hay checkpoint, usar motor Python con modelo sin entrenar
        # Permitir forzar el motor desde args (para cambio dinámico)
        logging.info(f"📋 Paso 5/10: Decidiendo motor a usar...")
        force_engine = args.get('force_engine', None)  # 'native', 'python', o None para auto
        if force_engine == 'python':
            use_native_engine = False
        elif force_engine == 'native':
            use_native_engine = True
        else:
            use_native_engine = getattr(global_cfg, 'USE_NATIVE_ENGINE', True)  # Por defecto True
        has_checkpoint = checkpoint_path is not None
        
        motor = None
        is_native = False
        
        if use_native_engine and has_checkpoint:
            # Verificar si el módulo nativo está disponible antes de intentar importarlo
            try:
                # Intentar importar el módulo nativo para verificar disponibilidad
                import atheria_core
                native_module_available = True
            except (ImportError, OSError, RuntimeError) as native_import_error:
                native_module_available = False
                # Solo loguear como debug, no como warning, porque es esperado si no está compilado
                logging.debug(f"Módulo nativo atheria_core no disponible: {native_import_error}. Usando motor Python.")
            
            if native_module_available:
                try:
                    from ..engines.native_engine_wrapper import NativeEngineWrapper
                    
                    # Buscar modelo JIT (exportado a TorchScript)
                    from ..utils import get_latest_jit_model
                    jit_path = get_latest_jit_model(exp_name, silent=True)
                    
                    # Si no existe modelo JIT, exportarlo automáticamente desde el checkpoint
                    if not jit_path:
                        logging.info(f"📋 Paso 6/10: Modelo JIT no encontrado para '{exp_name}'. Exportando automáticamente...")
                        if ws: await send_notification(ws, f"📦 Exportando modelo a TorchScript...", "info")
                        
                        # device_str ya está definido al inicio de la función
                        device = torch.device(device_str)
                        
                        try:
                            # MEJORA: Usar función mejorada de test_native_engine.py que maneja mejor
                            # el tamaño completo del grid y modelos ConvLSTM
                            import sys
                            import importlib.util
                            from pathlib import Path
                            
                            # Obtener el directorio raíz del proyecto
                            project_root = Path(__file__).parent.parent.parent
                            scripts_dir = project_root / "scripts"
                            test_native_path = scripts_dir / "test_native_engine.py"
                            
                            if not test_native_path.exists():
                                raise ImportError(f"No se encontró test_native_engine.py en {scripts_dir}")
                            
                            # Agregar el directorio scripts al path para que las importaciones funcionen
                            if str(scripts_dir) not in sys.path:
                                sys.path.insert(0, str(scripts_dir))
                            
                            # Cargar módulo dinámicamente
                            spec = importlib.util.spec_from_file_location("test_native_engine", test_native_path)
                            test_native_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(test_native_module)
                            export_model_to_torchscript = test_native_module.export_model_to_torchscript
                            
                            # El modelo ya está cargado (línea 936), usarlo directamente
                            # Asegurar que el modelo esté en modo evaluación y en el dispositivo correcto
                            model.eval()
                            model.to(device)
                            model.eval()
                            model.to(device)
                            
                            # Usar grid_size de inferencia (más grande que entrenamiento si aplica)
                            # Esto es importante para modelos UNet que necesitan el tamaño completo
                            export_grid_size = inference_grid_size
                            logging.info(f"Exportando modelo JIT usando device: {device_str}, grid_size: {export_grid_size}")
                            
                            # Exportar a JIT usando la función mejorada
                            jit_output_path = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name, "model_jit.pt")
                            exported_path = export_model_to_torchscript(
                                model,
                                device,
                                jit_output_path,
                                grid_size=export_grid_size,
                                d_state=d_state
                            )
                            
                            if exported_path and os.path.exists(exported_path):
                                jit_path = exported_path
                                logging.info(f"✅ Paso 6/10: Modelo exportado exitosamente a: {jit_path}")
                                if ws: await send_notification(ws, "✅ Modelo exportado a TorchScript", "success")
                            else:
                                logging.warning(f"⚠️ Error al exportar modelo JIT. Usando motor Python como fallback.")
                                if ws: await send_notification(ws, "⚠️ Error exportando a JIT, usando motor Python", "warning")
                                jit_path = None
                        except Exception as e:
                            logging.warning(f"⚠️ Error al exportar modelo JIT: {e}. Usando motor Python como fallback.", exc_info=True)
                            if ws: await send_notification(ws, f"⚠️ Error exportando JIT: {str(e)[:50]}...", "warning")
                            jit_path = None
                    
                    # Si tenemos modelo JIT, usar motor nativo
                    if jit_path and os.path.exists(jit_path):
                        logging.info(f"📋 Paso 7/10: Inicializando motor nativo...")
                        temp_motor = None
                        try:
                            # Usar auto-detección del device (configurado en config.py)
                            # Si device=None, usa auto-detección desde config.get_native_device()
                            temp_motor = NativeEngineWrapper(
                                grid_size=inference_grid_size,
                                d_state=d_state,
                                device=None,  # None = auto-detección desde config
                                cfg=config
                            )
                            logging.info(f"✅ Motor nativo inicializado con device: {temp_motor.device_str}")
                            
                            # Cargar modelo JIT en el motor nativo
                            logging.info(f"📋 Paso 8/10: Cargando modelo JIT en motor nativo...")
                            if temp_motor.load_model(jit_path):
                                motor = temp_motor
                                temp_motor = None  # Evitar cleanup - motor se usará
                                is_native = True
                                logging.info(f"✅ Paso 8/10: Modelo JIT cargado en motor nativo")
                                # Obtener versión del motor nativo después de cargar el modelo
                                try:
                                    if hasattr(motor, 'native_version'):
                                        native_version_loaded = motor.native_version
                                    else:
                                        native_version_loaded = "unknown"
                                except:
                                    native_version_loaded = "unknown"
                                logging.info(f"✅ Motor nativo (C++) cargado exitosamente con modelo JIT (version={native_version_loaded})")
                                if ws: await send_notification(ws, f"⚡ Motor nativo cargado (250-400x más rápido)", "success")
                            else:
                                logging.warning(f"⚠️ Error al cargar modelo JIT en motor nativo. Usando motor Python como fallback.")
                                if ws: await send_notification(ws, "⚠️ Error cargando modelo JIT, usando motor Python", "warning")
                                # Limpiar motor nativo que falló
                                if temp_motor is not None:
                                    try:
                                        if hasattr(temp_motor, 'cleanup'):
                                            temp_motor.cleanup()
                                    except Exception as cleanup_error:
                                        logging.debug(f"Error durante cleanup de motor nativo fallido: {cleanup_error}")
                                    temp_motor = None
                                motor = None
                        except Exception as e:
                            logging.warning(f"⚠️ Error al inicializar motor nativo: {e}. Usando motor Python como fallback.", exc_info=True)
                            if ws: await send_notification(ws, f"⚠️ Error en motor nativo, usando Python: {str(e)[:50]}...", "warning")
                            # CRÍTICO: Limpiar motor nativo que falló durante inicialización
                            if temp_motor is not None:
                                try:
                                    if hasattr(temp_motor, 'cleanup'):
                                        temp_motor.cleanup()
                                except Exception as cleanup_error:
                                    logging.debug(f"Error durante cleanup de motor nativo fallido: {cleanup_error}")
                                temp_motor = None
                            motor = None
                except Exception as e:
                    logging.warning(f"⚠️ Error en la inicialización del motor nativo: {e}. Usando motor Python como fallback.", exc_info=True)
                    if ws: await send_notification(ws, f"⚠️ Error en motor nativo, usando Python: {str(e)[:50]}...", "warning")
                    motor = None
            else:
                # Módulo nativo no disponible - usar motor Python directamente
                logging.debug(f"Módulo nativo no disponible, usando motor Python")
                motor = None
        
        # Fallback: usar motor Python tradicional
        # Esto se usa cuando:
        # 1. No hay checkpoint (modelo sin entrenar)
        # 2. Motor nativo no está disponible
        # 3. Error al cargar modelo JIT
        if motor is None:
            logging.info(f"📋 Paso 7/10: Usando motor Python (fallback)...")
            if not has_checkpoint:
                logging.info(f"Usando motor Python tradicional (Aetheria_Motor) - Modelo sin entrenar, iniciando con ruido aleatorio")
            else:
                logging.info(f"Usando motor Python tradicional (Aetheria_Motor)")
            # device_str ya está definido al inicio de la función
            # El estado inicial se creará automáticamente con ruido aleatorio según INITIAL_STATE_MODE_INFERENCE
            logging.info(f"📋 Paso 7.1/10: Creando instancia de Aetheria_Motor...")
            motor = Aetheria_Motor(model, inference_grid_size, d_state, global_cfg.DEVICE, cfg=config)
            logging.info(f"✅ Paso 7.1/10: Aetheria_Motor creado")
            
            # Compilar modelo para optimización de inferencia (solo para motor Python)
            try:
                motor.compile_model()
                if motor.is_compiled:
                    logging.info("✅ Modelo compilado con torch.compile() para inferencia optimizada")
                    if ws: await send_notification(ws, "✅ Modelo compilado con torch.compile() para mejor rendimiento", "info")
                else:
                    model_name = model.__class__.__name__
                    logging.info(f"ℹ️ torch.compile() deshabilitado para {model_name} (configuración del modelo)")
            except Exception as e:
                logging.warning(f"⚠️ No se pudo compilar el modelo: {e}. Continuando sin compilación.")
        
        # device_str ya está definido al inicio de la función
        
        g_state['motor'] = motor
        # CRÍTICO: Restaurar step desde checkpoint si está disponible
        # Si hay checkpoint, usar el step guardado; si no, empezar desde 0
        initial_step = checkpoint_step if checkpoint_path else 0
        g_state['simulation_step'] = initial_step
        g_state['initial_step'] = initial_step  # Guardar step inicial para mostrar "total - actual"
        g_state['checkpoint_step'] = checkpoint_step  # Step del checkpoint
        g_state['checkpoint_episode'] = checkpoint_episode  # Episode del checkpoint
        g_state['active_experiment'] = exp_name  # Guardar experimento activo
        
        # Información del grid para mostrar
        g_state['training_grid_size'] = training_grid_size
        g_state['inference_grid_size'] = inference_grid_size
        g_state['grid_size_ratio'] = inference_grid_size / training_grid_size if training_grid_size > 0 else 1.0
        
        # Inicializar FPS a 0.0 cuando se carga un experimento
        g_state['current_fps'] = 0.0
        g_state['fps_samples'] = []
        g_state['last_fps_calc_time'] = None
        
        # Guardar información sobre el tipo de motor para verificación
        motor_type = "native" if is_native else "python"
        g_state['motor_type'] = motor_type
        g_state['motor_is_native'] = is_native
        logging.info(f"✅ Motor almacenado en g_state: tipo={motor_type}, device={device_str}, is_native={is_native}")
        
        # Verificar que el motor tiene los métodos necesarios
        if hasattr(motor, 'evolve_internal_state'):
            logging.info(f"✅ Motor tiene método evolve_internal_state()")
        else:
            logging.error(f"❌ Motor NO tiene método evolve_internal_state()")
        
        # Verificar si es motor nativo
        if is_native and hasattr(motor, 'native_engine'):
            logging.info(f"✅ Motor nativo confirmado: tiene native_engine")
            logging.info(f"🚀 MOTOR NATIVO ACTIVO: device={device_str}, grid_size={inference_grid_size}")
        elif not is_native:
            logging.info(f"✅ Motor Python confirmado")
            logging.info(f"🐍 MOTOR PYTHON ACTIVO: device={device_str}, grid_size={inference_grid_size}")
        
        # Actualizar ROI manager con el tamaño correcto del grid
        from ..managers.roi_manager import ROIManager
        roi_manager = g_state.get('roi_manager')
        if roi_manager:
            roi_manager.grid_size = inference_grid_size
            roi_manager.clear_roi()  # Resetear ROI al cambiar de tamaño
        else:
            g_state['roi_manager'] = ROIManager(grid_size=inference_grid_size)
        
        # --- NOTA IMPORTANTE: La simulación queda en pausa después de cargar el modelo ---
        # Esto es INTENCIONAL: el modelo se carga en memoria y queda listo para ejecutar,
        # pero el usuario debe iniciarlo manualmente con 'play'.
        # Cargar modelo ≠ Ejecutar simulación (son operaciones separadas)
        g_state['is_paused'] = True
        g_state['live_feed_enabled'] = True  # Live feed habilitado por defecto al cargar modelo
        
        # Información del grid para mostrar en UI
        g_state['training_grid_size'] = training_grid_size
        g_state['inference_grid_size'] = inference_grid_size
        g_state['grid_size_ratio'] = inference_grid_size / training_grid_size if training_grid_size > 0 else 1.0
        
        # Enviar frame inicial inmediatamente para mostrar el estado inicial
        # SOLO si live_feed está habilitado
        logging.info(f"📋 Paso 9/10: Preparando frame inicial...")
        live_feed_enabled = g_state.get('live_feed_enabled', True)
        if live_feed_enabled:
            try:
                motor = g_state['motor']
                if motor:
                    # OPTIMIZACIÓN CRÍTICA: Usar lazy conversion para motor nativo
                    motor_is_native = g_state.get('motor_is_native', False)
                    if motor_is_native and hasattr(motor, 'get_dense_state'):
                        # Para motor nativo, usar get_dense_state() solo cuando se necesita visualizar
                        psi = motor.get_dense_state(check_pause_callback=lambda: g_state.get('is_paused', True))
                    else:
                        # Motor Python: acceder directamente (ya es denso)
                        psi = motor.state.psi if hasattr(motor, 'state') and motor.state else None
                    
                    if psi is not None:
                        delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                        viz_data = get_visualization_data(
                            psi, 
                            g_state.get('viz_type', 'density'),
                            delta_psi=delta_psi,
                            motor=motor
                        )
                    else:
                        viz_data = None
                    if viz_data and isinstance(viz_data, dict):
                        # Validar que los datos sean válidos antes de enviar
                        map_data = viz_data.get("map_data", [])
                        if map_data and len(map_data) > 0:
                            # Usar el step inicial (del checkpoint si existe, sino 0)
                            initial_step = g_state.get('initial_step', 0)
                            frame_payload_raw = {
                                "step": initial_step,
                                "timestamp": asyncio.get_event_loop().time(),
                                "map_data": map_data,
                                "hist_data": viz_data.get("hist_data", {}),
                                "poincare_coords": viz_data.get("poincare_coords", []),
                                "phase_attractor": viz_data.get("phase_attractor"),
                                "flow_data": viz_data.get("flow_data"),
                                "phase_hsv_data": viz_data.get("phase_hsv_data"),
                                "complex_3d_data": viz_data.get("complex_3d_data"),
                                "simulation_info": {
                                    "step": initial_step,
                                    "initial_step": initial_step,
                                    "checkpoint_step": checkpoint_step,
                                    "checkpoint_episode": checkpoint_episode,
                                    "is_paused": True,
                                    "live_feed_enabled": live_feed_enabled,
                                    "fps": g_state.get('current_fps', 0.0),
                                    "training_grid_size": training_grid_size,
                                    "inference_grid_size": inference_grid_size,
                                    "grid_scaled": training_grid_size != inference_grid_size
                                }
                            }
                            
                            # Aplicar optimizaciones si están habilitadas
                            compression_enabled = g_state.get('data_compression_enabled', True)
                            downsample_factor = g_state.get('downsample_factor', 1)
                            viz_type = g_state.get('viz_type', 'density')
                            
                            if compression_enabled or downsample_factor > 1:
                                frame_payload = await optimize_frame_payload(
                                    frame_payload_raw,
                                    enable_compression=compression_enabled,
                                    downsample_factor=downsample_factor,
                                    viz_type=viz_type
                                )
                            else:
                                frame_payload = frame_payload_raw
                            
                            # Verificar que el payload tenga step antes de enviar
                            initial_step = g_state.get('initial_step', 0)
                            if 'step' not in frame_payload:
                                logging.warning(f"⚠️ Frame inicial sin step, añadiendo step={initial_step}")
                                frame_payload['step'] = initial_step
                            
                            await broadcast({"type": "simulation_frame", "payload": frame_payload})
                            logging.info(f"Frame inicial enviado exitosamente para '{exp_name}' (step={initial_step}, keys={list(frame_payload.keys())})")
                        else:
                            logging.warning("get_visualization_data retornó map_data vacío.")
                            if ws: await send_notification(ws, "⚠️ Modelo cargado pero sin datos de visualización iniciales.", "warning")
                    else:
                        logging.warning("get_visualization_data retornó datos inválidos para frame inicial.")
                        if ws: await send_notification(ws, "⚠️ Error generando datos de visualización iniciales.", "warning")
                else:
                    logging.warning("Motor cargado pero sin estado psi inicial.")
                    if ws: await send_notification(ws, "⚠️ Modelo cargado pero el estado cuántico no se inicializó correctamente.", "warning")
            except Exception as e:
                logging.error(f"Error generando frame inicial: {e}", exc_info=True)
                if ws: await send_notification(ws, f"⚠️ Error al generar visualización inicial: {str(e)}", "warning")
        
        # Enviar información sobre el estado del motor
        logging.info(f"📋 Paso 10/10: Enviando estado final...")
        if is_native:
            # Obtener device del motor nativo
            device_str = motor.device_str if hasattr(motor, 'device_str') else 'cpu'
            
            # Obtener versiones del motor nativo
            # NativeEngineWrapper ya está importado en el bloque anterior si is_native es True
            try:
                from ..engines.native_engine_wrapper import NativeEngineWrapper as NativeEngineWrapperClass
                wrapper_version = getattr(NativeEngineWrapperClass, 'VERSION', None) or "unknown"
            except ImportError:
                wrapper_version = "unknown"
            
            native_version = getattr(motor, 'native_version', None) or "unknown"
            
            compile_status = {
                "is_compiled": True,  # Motor nativo siempre está "compilado"
                "is_native": True,
                "model_name": "Native Engine (C++)",
                "compiles_enabled": True,
                "device_str": device_str,  # CPU/CUDA - CORREGIDO: usar device_str en lugar de device
                "native_version": native_version,  # Versión del motor C++ (SemVer)
                "wrapper_version": wrapper_version  # Versión del wrapper Python (SemVer)
            }
            logging.info(f"📤 Enviando compile_status NATIVO: is_native=True, device_str={device_str}, native_version={native_version}, wrapper_version={wrapper_version}")
        else:
            # Motor Python: obtener device del motor o usar global
            device_str = str(motor.device) if hasattr(motor, 'device') else str(global_cfg.DEVICE)
            # Extraer solo 'cpu' o 'cuda' del device string
            if 'cuda' in device_str.lower():
                device_str = 'cuda'
            else:
                device_str = 'cpu'
            
            # Obtener versión del motor Python
            python_version = getattr(motor, 'VERSION', None) or (motor.get_version() if hasattr(motor, 'get_version') else 'unknown')
            
            compile_status = {
                "is_compiled": motor.is_compiled,
                "is_native": False,
                "model_name": model.__class__.__name__ if hasattr(model, '__class__') else "Unknown",
                "compiles_enabled": getattr(model, '_compiles', True) if hasattr(model, '_compiles') else True,
                "device_str": device_str,  # CPU/CUDA - CORREGIDO: usar device_str en lugar de device
                "python_version": python_version  # Versión del motor Python (SemVer)
            }
            logging.info(f"📤 Enviando compile_status PYTHON: is_native=False, device_str={device_str}, python_version={python_version}")
        
        # Logging detallado para debugging
        logging.info(f"📊 compile_status completo: {compile_status}")
        
        if ws: await send_notification(ws, f"✅ Modelo '{exp_name}' cargado exitosamente. Presiona 'Iniciar' para comenzar la simulación.", "success")
        
        # Enviar compile_status en el broadcast con información del checkpoint y grid
        initial_step = checkpoint_step if checkpoint_path else 0
        status_payload = {
            "status": "paused",
            "model_loaded": True,
            "experiment_name": exp_name,
            "compile_status": compile_status,
            "checkpoint_info": {
                "checkpoint_path": checkpoint_path if checkpoint_path else None,
                "checkpoint_step": checkpoint_step,
                "checkpoint_episode": checkpoint_episode,
                "initial_step": initial_step,
                "training_grid_size": training_grid_size,
                "inference_grid_size": inference_grid_size,
                "grid_scaled": training_grid_size != inference_grid_size,
                "grid_size_ratio": inference_grid_size / training_grid_size if training_grid_size > 0 else 1.0
            }
        }
        logging.info(f"📤 Enviando inference_status_update con compile_status: {status_payload}")
        await broadcast({
            "type": "inference_status_update", 
            "payload": status_payload
        })
        
        # Mensaje informativo sobre el checkpoint y grid
        if checkpoint_path:
            grid_msg = f" (Grid escalado: {training_grid_size}→{inference_grid_size})" if training_grid_size != inference_grid_size else ""
            logging.info(f"Modelo '{exp_name}' cargado desde checkpoint (episode={checkpoint_episode}, step={checkpoint_step}){grid_msg}. Simulación en pausa, esperando inicio manual.")
        else:
            logging.info(f"Modelo '{exp_name}' cargado sin checkpoint (nuevo modelo). Simulación en pausa, esperando inicio manual.")
        
        # Logging adicional para verificación (ya se hizo arriba, pero para confirmar)
        if is_native:
            logging.info(f"🚀 MOTOR NATIVO LISTO: device={device_str}, grid_size={inference_grid_size}")
        else:
            logging.info(f"🐍 MOTOR PYTHON LISTO: device={device_str}, grid_size={inference_grid_size}")

    except Exception as e:
        logging.error(f"Error crítico cargando experimento '{exp_name}': {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al cargar '{exp_name}': {str(e)}", "error")

async def handle_switch_engine(args):
    """Cambia entre motor nativo (C++) y motor Python si están disponibles."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    target_engine = args.get('engine', 'auto')  # 'native', 'python', o 'auto'
    
    motor = g_state.get('motor')
    # Permitir cambiar de motor incluso sin modelo cargado
    # Si no hay modelo, simplemente guardar la preferencia
    # if not motor:
    #     if ws: await send_notification(ws, "⚠️ No hay modelo cargado. Carga un experimento primero.", "warning")
    #     return
    
    # Verificar qué motor está actualmente en uso
    current_is_native = hasattr(motor, 'native_engine') if motor else False
    current_engine_type = 'native' if current_is_native else 'python'
    
    # Determinar qué motor usar
    if target_engine == 'auto':
        # Auto: cambiar al opuesto si es posible
        target_engine = 'python' if current_is_native else 'native'
    elif target_engine == current_engine_type:
        if ws: await send_notification(ws, f"⚠️ Ya estás usando el motor {current_engine_type}.", "info")
        return
    
    # Obtener información del experimento actual
    # Si no hay experimento cargado, simplemente guardar la preferencia para cuando se cargue uno
    exp_name = g_state.get('active_experiment')
    if not exp_name:
        # Guardar preferencia de motor sin modelo
        if target_engine == 'native':
            # Verificar si el motor nativo está disponible
            try:
                import atheria_core
                if ws: await send_notification(ws, "✅ Motor nativo seleccionado. Se usará cuando cargues un experimento con checkpoint.", "info")
            except (ImportError, OSError, RuntimeError):
                if ws: await send_notification(ws, "⚠️ Motor nativo no disponible. El módulo C++ no está compilado.", "error")
        else:
            if ws: await send_notification(ws, "✅ Motor Python seleccionado. Se usará cuando cargues un experimento.", "info")
        return
    
    # Verificar disponibilidad del motor objetivo
    if target_engine == 'native':
        try:
            import atheria_core
            native_available = True
        except (ImportError, OSError, RuntimeError):
            native_available = False
            if ws: await send_notification(ws, "⚠️ Motor nativo no disponible. El módulo C++ no está compilado.", "error")
            return
        
        # Verificar que haya checkpoint y modelo JIT
        from ..utils import get_latest_checkpoint, get_latest_jit_model
        checkpoint_path = get_latest_checkpoint(exp_name)
        jit_path = get_latest_jit_model(exp_name, silent=True)
        
        if not checkpoint_path:
            if ws: await send_notification(ws, "⚠️ Motor nativo requiere un modelo entrenado (checkpoint).", "error")
            return
        
        if not jit_path:
            if ws: await send_notification(ws, "📦 Modelo JIT no encontrado. Exportando automáticamente...", "info")
            # El export se hará en handle_load_experiment
            # Por ahora, simplemente recargar el experimento con motor nativo
            await handle_load_experiment({
                'ws_id': args.get('ws_id'),
                'experiment_name': exp_name,
                'force_engine': 'native'
            })
            return
    
    # Pausar simulación si está corriendo
    was_running = not g_state.get('is_paused', True)
    if was_running:
        g_state['is_paused'] = True
        await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
    
    # CRÍTICO: Limpiar motor anterior antes de cambiar para prevenir segfaults
    old_motor = motor
    try:
        # Limpiar motor nativo explícitamente antes de eliminarlo
        if old_motor and hasattr(old_motor, 'native_engine'):
            try:
                if hasattr(old_motor, 'cleanup'):
                    old_motor.cleanup()
                    logging.debug("Motor nativo limpiado explícitamente antes de cambiar de engine")
                else:
                    # Fallback: limpiar manualmente
                    if hasattr(old_motor, 'native_engine') and old_motor.native_engine is not None:
                        old_motor.native_engine = None
                    if hasattr(old_motor, 'state') and old_motor.state is not None:
                        if hasattr(old_motor.state, 'psi') and old_motor.state.psi is not None:
                            old_motor.state.psi = None
                        old_motor.state = None
            except Exception as cleanup_error:
                logging.warning(f"Error durante cleanup de motor nativo al cambiar engine: {cleanup_error}")
    except Exception as e:
        logging.warning(f"Error limpiando motor anterior al cambiar engine: {e}", exc_info=True)
    
    # Guardar estado actual si es posible
    current_step = g_state.get('simulation_step', 0)
    current_psi = None
    try:
        if motor and hasattr(motor, 'state') and motor.state and motor.state.psi is not None:
            current_psi = motor.state.psi.clone().detach()
    except Exception as e:
        logging.warning(f"Error guardando estado al cambiar engine: {e}")
        current_psi = None
    
    try:
        # Recargar el experimento con el motor objetivo
        # Usar handle_load_experiment pero forzar el motor específico
        await handle_load_experiment({
            'ws_id': args.get('ws_id'),
            'experiment_name': exp_name,
            'force_engine': target_engine
        })
        
        # Restaurar el paso y estado si es posible
        if current_psi is not None:
            new_motor = g_state.get('motor')
            if new_motor and hasattr(new_motor, 'state'):
                try:
                    # Intentar restaurar el estado (puede fallar si cambió el formato)
                    new_motor.state.psi = current_psi.to(new_motor.state.psi.device)
                    g_state['simulation_step'] = current_step
                    logging.info(f"✅ Estado restaurado al cambiar de motor (step={current_step})")
                except Exception as e:
                    logging.warning(f"⚠️ No se pudo restaurar el estado: {e}. Usando estado inicial.")
        
        if ws: 
            engine_label = "Nativo (C++)" if target_engine == 'native' else "Python"
            await send_notification(ws, f"✅ Cambiado a motor {engine_label}", "success")
        
        # Reanudar simulación si estaba corriendo
        if was_running:
            await handle_play({'ws_id': args.get('ws_id')})
            
    except Exception as e:
        logging.error(f"Error cambiando de motor: {e}", exc_info=True)
        if ws: await send_notification(ws, f"❌ Error cambiando de motor: {str(e)[:50]}...", "error")

async def handle_unload_model(args):
    """Descarga el modelo cargado y limpia el estado."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    motor = g_state.get('motor')
    
    if not motor:
        if ws: 
            try:
                await send_notification(ws, "⚠️ No hay modelo cargado para descargar.", "warning")
            except (ConnectionResetError, ConnectionError, OSError):
                pass  # Cliente ya desconectado, ignorar
        return
    
    try:
        # Limpiar motor y estado
        experiment_name = g_state.get('active_experiment', 'Unknown')
        
        # CRÍTICO: Limpiar motor nativo correctamente antes de eliminarlo
        if hasattr(motor, 'native_engine') and motor.native_engine is not None:
            # Motor nativo: usar método cleanup() si existe
            try:
                if hasattr(motor, 'cleanup'):
                    motor.cleanup()
                    logging.debug("Motor nativo limpiado usando método cleanup()")
                elif hasattr(motor, 'native_engine') and hasattr(motor.native_engine, 'clear'):
                    motor.native_engine.clear()
                    logging.debug("Motor nativo limpiado usando clear()")
            except Exception as cleanup_error:
                logging.warning(f"Error durante cleanup de motor nativo (no crítico): {cleanup_error}")
        
        # Limpiar estado del motor
        if hasattr(motor, 'state') and motor.state is not None:
            try:
                if hasattr(motor.state, 'psi') and motor.state.psi is not None:
                    del motor.state.psi
                    motor.state.psi = None
                del motor.state
                motor.state = None
            except Exception as state_cleanup_error:
                logging.debug(f"Error limpiando estado del motor (no crítico): {state_cleanup_error}")
        
        # Limpiar referencia del motor (no usar del directamente)
        motor = None
        
        # Limpiar g_state ANTES de eliminar el motor
        g_state['motor'] = None
        g_state['simulation_step'] = 0
        g_state['motor_type'] = None
        g_state['motor_is_native'] = False
        g_state['active_experiment'] = None
        g_state['is_paused'] = True
        
        # Limpiar snapshots y otros datos
        if 'snapshots' in g_state:
            try:
                g_state['snapshots'].clear()
            except Exception:
                pass
        
        if 'simulation_history' in g_state:
            try:
                g_state['simulation_history'].clear()
            except Exception:
                pass
        
        # Limpiar cache de CUDA si está disponible
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("Cache de CUDA limpiado")
        except Exception as cuda_cleanup_error:
            logging.debug(f"Error limpiando cache de CUDA (no crítico): {cuda_cleanup_error}")
        
        # Forzar garbage collection
        import gc
        gc.collect()
        
        logging.info(f"✅ Modelo '{experiment_name}' descargado y memoria limpiada")
        
        # Enviar notificación solo si el WebSocket sigue conectado
        if ws and not ws.closed:
            try:
                await send_notification(ws, f"✅ Modelo descargado. Memoria limpiada.", "success")
            except (ConnectionResetError, ConnectionError, OSError):
                logging.debug("WebSocket cerrado durante notificación de descarga (normal)")
        
        # Enviar actualización de estado a todos los clientes conectados
        try:
            await broadcast({
                "type": "inference_status_update",
                "payload": {
                    "status": "paused",
                    "model_loaded": False,
                    "experiment_name": None,
                    "compile_status": None
                }
            })
        except Exception as broadcast_error:
            logging.warning(f"Error enviando broadcast de estado después de descarga: {broadcast_error}")
        
    except Exception as e:
        logging.error(f"Error descargando modelo: {e}", exc_info=True)
        # Enviar notificación solo si el WebSocket sigue conectado
        if ws and not ws.closed:
            try:
                await send_notification(ws, f"⚠️ Error al descargar modelo: {str(e)[:50]}...", "error")
            except (ConnectionResetError, ConnectionError, OSError):
                pass  # Cliente ya desconectado, ignorar

async def handle_reset(args):
    """Reinicia el estado de la simulación al estado inicial."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    motor = g_state.get('motor')
    
    if not motor:
        msg = "⚠️ No hay modelo cargado. Carga un experimento primero."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    try:
        # Obtener el modo de inicialización de la configuración
        from ..utils import load_experiment_config
        
        # Intentar obtener el modo de inicialización del experimento activo o usar el global
        initial_mode = getattr(global_cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
        
        # Reiniciar el estado cuántico con el modo de inicialización correcto
        motor.state = QuantumState(
            motor.grid_size, 
            motor.d_state, 
            motor.device,
            initial_mode=initial_mode
        )
        g_state['simulation_step'] = 0
        
        # Enviar frame actualizado si live_feed está habilitado
        live_feed_enabled = g_state.get('live_feed_enabled', True)
        if live_feed_enabled:
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
                        frame_payload_raw = {
                            "step": 0,
                            "timestamp": asyncio.get_event_loop().time(),
                            "map_data": map_data,
                            "hist_data": viz_data.get("hist_data", {}),
                            "poincare_coords": viz_data.get("poincare_coords", []),
                            "phase_attractor": viz_data.get("phase_attractor"),
                            "flow_data": viz_data.get("flow_data"),
                            "phase_hsv_data": viz_data.get("phase_hsv_data"),
                            "complex_3d_data": viz_data.get("complex_3d_data"),
                            "simulation_info": {
                                "step": 0,
                                "is_paused": True,
                                "live_feed_enabled": live_feed_enabled,
                                "fps": g_state.get('current_fps', 0.0)
                            }
                        }
                        
                        # Aplicar optimizaciones si están habilitadas
                        compression_enabled = g_state.get('data_compression_enabled', True)
                        downsample_factor = g_state.get('downsample_factor', 1)
                        viz_type = g_state.get('viz_type', 'density')
                        
                        if compression_enabled or downsample_factor > 1:
                            frame_payload = await optimize_frame_payload(
                                frame_payload_raw,
                                enable_compression=compression_enabled,
                                downsample_factor=downsample_factor,
                                viz_type=viz_type
                            )
                        else:
                            frame_payload = frame_payload_raw
                        
                        # Verificar que el payload tenga step antes de enviar
                        if 'step' not in frame_payload:
                            logging.warning(f"⚠️ Frame de reinicio sin step, añadiendo step=0")
                            frame_payload['step'] = 0
                        
                        await broadcast({"type": "simulation_frame", "payload": frame_payload})
                        logging.info(f"Frame de reinicio enviado exitosamente (step=0, keys={list(frame_payload.keys())})")
                    else:
                        logging.warning("get_visualization_data retornó map_data vacío al reiniciar")
                else:
                    logging.warning("get_visualization_data retornó datos inválidos al reiniciar")
            except Exception as e:
                logging.error(f"Error generando frame de reinicio: {e}", exc_info=True)
        
        msg = f"✅ Estado de simulación reiniciado (modo: {initial_mode})."
        if ws: await send_notification(ws, msg, "success")
        logging.info(f"Simulación reiniciada por [{args.get('ws_id')}]")
        
    except Exception as e:
        logging.error(f"Error al reiniciar simulación: {e}", exc_info=True)
        msg = f"❌ Error al reiniciar: {str(e)}"
        if ws: await send_notification(ws, msg, "error")

async def handle_shutdown(args):
    """
    Handler para apagar el servidor desde la UI.
    
    Args:
        args: Dict con parámetros (opcional: 'confirm'=True)
    """
    ws = args.get('ws') if isinstance(args, dict) else None
    
    try:
        # Verificar confirmación
        confirm = args.get('confirm', False) if isinstance(args, dict) else False
        if not confirm:
            if ws:
                await send_notification(ws, "⚠️ Shutdown requiere confirmación. Envía con confirm=true", "warning")
            return
        
        # Notificar a todos los clientes que el servidor se apagará
        await broadcast({
            "type": "server_shutdown",
            "message": "Servidor apagándose en 2 segundos..."
        })
        
        # Esperar un momento para que el mensaje se envíe
        await asyncio.sleep(0.5)
        
        # Activar shutdown event si está disponible
        shutdown_event = g_state.get('shutdown_event')
        if shutdown_event:
            shutdown_event.set()
            logging.info("🚀 Shutdown solicitado desde UI. Evento activado.")
            if ws:
                await send_notification(ws, "✅ Comando de shutdown enviado", "success")
        else:
            # Fallback: usar os._exit si no hay evento
            logging.warning("⚠️ Shutdown event no disponible. Usando os._exit()")
            import os
            if ws:
                await send_notification(ws, "⚠️ Apagando servidor...", "warning")
            # Esperar un momento antes de forzar salida
            await asyncio.sleep(1.0)
            os._exit(0)
            
    except Exception as e:
        logging.error(f"Error en handle_shutdown: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al apagar servidor: {str(e)}", "error")

async def handle_refresh_experiments(args):
    """Refresca la lista de experimentos y la envía a todos los clientes conectados."""
    try:
        experiments = get_experiment_list()
        await broadcast({
            "type": "experiments_updated",
            "payload": {
                "experiments": experiments
            }
        })
        logging.info(f"Lista de experimentos actualizada y enviada a clientes ({len(experiments)} experimentos)")
    except Exception as e:
        logging.error(f"Error al refrescar lista de experimentos: {e}", exc_info=True)

async def handle_list_checkpoints(args):
    """Lista todos los checkpoints de un experimento."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    
    if not exp_name:
        if ws: await send_notification(ws, "Nombre de experimento no proporcionado.", "error")
        return
    
    try:
        import os
        import glob
        from datetime import datetime
        
        checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        if not os.path.exists(checkpoint_dir):
            if ws: await send_to_websocket(ws, "checkpoints_list", {"checkpoints": []})
            return
        
        checkpoints = []
        for checkpoint_file in glob.glob(os.path.join(checkpoint_dir, "*.pth")):
            try:
                stat = os.stat(checkpoint_file)
                filename = os.path.basename(checkpoint_file)
                
                # Extraer episodio del nombre
                episode = 0
                is_best = 'best' in filename.lower()
                if 'ep' in filename:
                    try:
                        episode = int(filename.split('ep')[-1].split('.')[0])
                    except:
                        pass
                
                checkpoints.append({
                    "filename": filename,
                    "episode": episode,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_best": is_best
                })
            except Exception as e:
                logging.warning(f"Error procesando checkpoint {checkpoint_file}: {e}")
        
        # Ordenar por episodio
        checkpoints.sort(key=lambda x: x['episode'], reverse=True)
        
        if ws: await send_to_websocket(ws, "checkpoints_list", {"checkpoints": checkpoints})
        
    except Exception as e:
        logging.error(f"Error al listar checkpoints: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al listar checkpoints: {str(e)}", "error")

async def handle_cleanup_checkpoints(args):
    """Limpia checkpoints antiguos de un experimento, manteniendo los N más recientes y el mejor."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get('EXPERIMENT_NAME')
    
    if not exp_name:
        if ws: await send_notification(ws, "Nombre de experimento no proporcionado.", "error")
        return
        
    try:
        checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        if not os.path.exists(checkpoint_dir):
            if ws: await send_notification(ws, f"Directorio de checkpoints no encontrado para {exp_name}", "warning")
            return
            
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            if f.startswith("checkpoint_ep") and f.endswith(".pth"):
                full_path = os.path.join(checkpoint_dir, f)
                try:
                    # Extraer número de episodio del nombre
                    ep_num = int(f.replace("checkpoint_ep", "").replace(".pth", ""))
                    checkpoints.append((ep_num, full_path))
                except ValueError:
                    continue
        
        # Ordenar por episodio (descendente)
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Verificar si hay un checkpoint "best"
        best_checkpoint = None
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        if os.path.exists(best_path):
            best_checkpoint = best_path
        
        # Mantener los últimos 5 checkpoints + el mejor
        max_keep = 5
        deleted_count = 0
        
        if len(checkpoints) > max_keep:
            to_delete = checkpoints[max_keep:]
            
            for ep, path in to_delete:
                if path != best_checkpoint:
                    try:
                        os.remove(path)
                        deleted_count += 1
                        logging.info(f"Checkpoint eliminado: {path}")
                    except Exception as e:
                        logging.warning(f"Error eliminando {path}: {e}")
        
        msg = f"✅ Limpieza completada. {deleted_count} checkpoints eliminados."
        if ws: await send_notification(ws, msg, "success")
        
        # Actualizar lista de checkpoints
        await handle_list_checkpoints({'ws_id': args.get('ws_id'), 'EXPERIMENT_NAME': exp_name})
        
    except Exception as e:
        logging.error(f"Error en limpieza de checkpoints: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al limpiar checkpoints: {str(e)}", "error")

async def handle_delete_checkpoint(args):
    """Elimina un checkpoint específico."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    checkpoint_name = args.get("CHECKPOINT_NAME")
    
    if not exp_name or not checkpoint_name:
        if ws: await send_notification(ws, "Faltan parámetros requeridos.", "error")
        return
    
    try:
        import os
        import shutil
        checkpoint_path = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            if ws: await send_notification(ws, f"Checkpoint '{checkpoint_name}' no encontrado.", "error")
            return
        
        os.remove(checkpoint_path)
        if ws: await send_notification(ws, f"✅ Checkpoint '{checkpoint_name}' eliminado.", "success")
        
        # Enviar lista actualizada
        await handle_list_checkpoints(args)
        
    except Exception as e:
        logging.error(f"Error al eliminar checkpoint: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al eliminar checkpoint: {str(e)}", "error")

async def handle_delete_experiment(args):
    """Elimina un experimento completo (configuración y checkpoints)."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    
    if not exp_name:
        if ws: await send_notification(ws, "Nombre de experimento no proporcionado.", "error")
        return
    
    try:
        import os
        import shutil
        
        # Eliminar directorio del experimento (config.json)
        exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, exp_name)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            logging.info(f"Directorio de experimento eliminado: {exp_dir}")
        
        # Eliminar directorio de checkpoints
        checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logging.info(f"Directorio de checkpoints eliminado: {checkpoint_dir}")
        
        # Si el experimento activo era el eliminado, limpiar el estado
        active_exp = g_state.get('active_experiment')
        if g_state.get('motor') and active_exp:
            if active_exp == exp_name:
                g_state['motor'] = None
                g_state['active_experiment'] = None
                g_state['simulation_step'] = 0
                g_state['is_paused'] = True
                await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
        
        if ws: await send_notification(ws, f"✅ Experiment '{exp_name}' eliminado exitosamente.", "success")
        
        # Enviar lista actualizada de experimentos
        await handle_refresh_experiments(args)
        
    except Exception as e:
        logging.error(f"Error al eliminar experimento '{exp_name}': {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al eliminar experimento: {str(e)}", "error")

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
        import threading
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
        import concurrent.futures
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
        import threading
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
        import concurrent.futures
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
    """Cancela un análisis en curso."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    if g_state.get('analysis_status') != 'running':
        msg = "⚠️ No hay ningún análisis en curso."
        if ws:
            await send_notification(ws, msg, "warning")
        return
    
    try:
        logging.info("Cancelando análisis en curso...")
        
        # Cancelar tarea
        if g_state.get('analysis_task'):
            g_state['analysis_task'].cancel()
        
        # Señalar cancelación
        if g_state.get('analysis_cancel_event'):
            g_state['analysis_cancel_event'].set()
        
        # Actualizar estado
        analysis_type = g_state.get('analysis_type')
        g_state['analysis_status'] = 'idle'
        g_state['analysis_type'] = None
        
        # Notificar
        await broadcast({
            "type": "analysis_status_update",
            "payload": {"status": "cancelled", "type": None}
        })
        
        msg = f"✅ Análisis {analysis_type} cancelado."
        if ws:
            await send_notification(ws, msg, "info")
        logging.info(msg)
        
    except Exception as e:
        logging.error(f"Error al cancelar análisis: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"❌ Error al cancelar análisis: {str(e)}", "error")

async def handle_clear_snapshots(args):
    """Limpia los snapshots almacenados."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    try:
        n_before = len(g_state.get('snapshots', []))
        g_state['snapshots'] = []
        
        msg = f"✅ Snapshots limpiados ({n_before} eliminados)"
        logging.info(msg)
        if ws:
            await send_notification(ws, msg, "success")
        
    except Exception as e:
        logging.error(f"Error al limpiar snapshots: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al limpiar snapshots: {str(e)}", "error")

async def handle_set_steps_interval(args):
    """Configura el intervalo de pasos para mostrar frames cuando live feed está desactivado.
    
    Args:
        steps_interval: Intervalo en pasos. Valores:
            - -1: Fullspeed (no enviar frames automáticamente, máxima velocidad)
            - 0: Modo manual (solo actualizar con botón)
            - 1-1000000: Enviar frame cada N pasos automáticamente (permite intervalos muy grandes)
    """
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    try:
        steps_interval = args.get('steps_interval', 10)
        if steps_interval == -1:
            # Modo fullspeed: no enviar frames automáticamente
            g_state['steps_interval'] = -1
            g_state['steps_interval_counter'] = 0  # Resetear contador
            msg = "✅ Modo fullspeed activado: simulación a máxima velocidad sin enviar frames"
        elif steps_interval < 0:
            steps_interval = -1  # Permitir -1 para modo fullspeed
            g_state['steps_interval'] = -1
            g_state['steps_interval_counter'] = 0
            msg = "✅ Modo fullspeed activado: simulación a máxima velocidad sin enviar frames"
        elif steps_interval == 0:
            # Modo manual: solo actualizar con botón
            g_state['steps_interval'] = 0
            g_state['steps_interval_counter'] = 0
            msg = "✅ Modo manual activado: solo actualizar con botón 'Actualizar Visualización'"
        elif steps_interval > 1000000:  # Límite aumentado a 1 millón
            steps_interval = 1000000
            logging.warning(f"steps_interval limitado a 1,000,000 (valor solicitado: {args.get('steps_interval')})")
            g_state['steps_interval'] = int(steps_interval)
            g_state['steps_interval_counter'] = 0
            steps_str = f"{steps_interval:,}".replace(",", ".")
            msg = f"✅ Intervalo de pasos configurado: cada {steps_str} pasos"
        else:
            g_state['steps_interval'] = int(steps_interval)
            g_state['steps_interval_counter'] = 0
            # Formatear número con separadores de miles para mejor legibilidad
            steps_str = f"{steps_interval:,}".replace(",", ".")
            msg = f"✅ Intervalo de pasos configurado: cada {steps_str} pasos"
        
        logging.info(msg)
        if ws:
            await send_notification(ws, msg, "success")
    except Exception as e:
        logging.error(f"Error al configurar intervalo de pasos: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al configurar intervalo: {str(e)}", "error")

async def handle_set_snapshot_interval(args):
    """Configura el intervalo de captura de snapshots."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    try:
        interval = args.get('interval', 500)
        if interval < 1:
            interval = 1
        
        g_state['snapshot_interval'] = interval
        
        msg = f"✅ Intervalo de snapshots configurado: cada {interval} pasos"
        logging.info(msg)
        if ws:
            await send_notification(ws, msg, "success")
        
    except Exception as e:
        logging.error(f"Error al configurar intervalo de snapshots: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al configurar intervalo: {str(e)}", "error")

async def handle_enable_snapshots(args):
    """Habilita o deshabilita la captura de snapshots."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    try:
        enabled = args.get('enabled', True)
        g_state['snapshot_enabled'] = enabled
        
        msg = f"✅ Captura de snapshots {'habilitada' if enabled else 'deshabilitada'}"
        logging.info(msg)
        if ws:
            await send_notification(ws, msg, "success")
        
    except Exception as e:
        logging.error(f"Error al configurar snapshots: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al configurar snapshots: {str(e)}", "error")

async def handle_set_live_feed(args):
    """Habilita o deshabilita el envío de datos en tiempo real (live feed)."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    enabled = args.get('enabled', True)
    g_state['live_feed_enabled'] = bool(enabled)
    
    # Resetear contador de pasos cuando se activa/desactiva live feed
    if 'steps_interval_counter' not in g_state:
        g_state['steps_interval_counter'] = 0
    if 'last_frame_sent_step' not in g_state:
        g_state['last_frame_sent_step'] = -1  # Forzar envío de frame inicial
    
    status_msg = "activado" if enabled else "desactivado"
    logging.info(f"Live feed {status_msg}. La simulación continuará ejecutándose {'sin calcular visualizaciones' if not enabled else 'y enviando datos en tiempo real'}.")
    
    if ws:
        await send_notification(ws, f"Live feed {status_msg}. {'La simulación corre sin calcular visualizaciones para mejor rendimiento.' if not enabled else 'Enviando datos en tiempo real.'}", "info")
    
    # Si se desactiva live feed, enviar un frame inicial inmediatamente para que el usuario vea algo
    if not enabled and g_state.get('motor') and not g_state.get('is_paused', True):
        try:
            current_step = g_state.get('simulation_step', 0)
            delta_psi = g_state['motor'].last_delta_psi if hasattr(g_state['motor'], 'last_delta_psi') else None
            viz_data = get_visualization_data(
                g_state['motor'].state.psi, 
                g_state.get('viz_type', 'density'),
                delta_psi=delta_psi,
                motor=g_state['motor']
            )
            
            if viz_data and isinstance(viz_data, dict):
                map_data = viz_data.get("map_data", [])
                if map_data and len(map_data) > 0:
                    frame_payload_raw = {
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
                            "live_feed_enabled": False,
                            "fps": g_state.get('current_fps', 0.0)
                        }
                    }
                    
                    # Aplicar optimizaciones si están habilitadas
                    compression_enabled = g_state.get('data_compression_enabled', True)
                    downsample_factor = g_state.get('downsample_factor', 1)
                    
                    if compression_enabled or downsample_factor > 1:
                        frame_payload = await optimize_frame_payload(
                            frame_payload_raw,
                            enable_compression=compression_enabled,
                            downsample_factor=downsample_factor,
                            viz_type=g_state.get('viz_type', 'density')
                        )
                    else:
                        frame_payload = frame_payload_raw
                    
                    await broadcast({"type": "simulation_frame", "payload": frame_payload})
                    g_state['last_frame_sent_step'] = current_step
                    logging.info(f"Frame inicial enviado al desactivar live feed (paso {current_step})")
        except Exception as e:
            logging.warning(f"Error enviando frame inicial al desactivar live feed: {e}")
    
    # Enviar actualización a todos los clientes
    await broadcast({
        "type": "live_feed_status_update",
        "payload": {"enabled": g_state['live_feed_enabled']}
    })

async def handle_set_compression(args):
    """Habilita o deshabilita la compresión de datos WebSocket."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    enabled = args.get('enabled', True)
    g_state['data_compression_enabled'] = bool(enabled)
    
    status_msg = "activada" if enabled else "desactivada"
    logging.info(f"Compresión de datos {status_msg}.")
    
    if ws:
        await send_notification(ws, f"Compresión {status_msg}. {'Datos optimizados para transferencia.' if enabled else 'Datos sin comprimir.'}", "info")
    
    await broadcast({
        "type": "compression_status_update",
        "payload": {"enabled": g_state['data_compression_enabled']}
    })

async def handle_set_downsample(args):
    """Configura el factor de downsampling para transferencia de datos."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    factor = args.get("factor", 1)
    if factor < 1:
        factor = 1
    elif factor > 8:
        factor = 8
    g_state['downsample_factor'] = int(factor)
    
    logging.info(f"Downsampling ajustado a: {factor}x ({'sin downsampling' if factor == 1 else f'{factor}x reducción'})")
    
    if ws:
        if factor == 1:
            await send_notification(ws, "Downsampling desactivado. Enviando datos a resolución completa.", "info")
        else:
            grid_size = g_state.get('roi_manager', None)
            grid_size = grid_size.grid_size if grid_size else 256
            await send_notification(ws, f"Downsampling activado: {factor}x reducción (resolución {grid_size//factor}x{grid_size//factor})", "info")
    
    await broadcast({
        "type": "downsample_status_update",
        "payload": {"factor": g_state['downsample_factor']}
    })

async def handle_set_roi(args):
    """Configura la región de interés (ROI) para visualización."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    roi_manager = g_state.get('roi_manager')
    
    if not roi_manager:
        # Crear ROI manager si no existe
        from ..managers.roi_manager import ROIManager
        grid_size = g_state.get('grid_size', 256)
        roi_manager = ROIManager(grid_size=grid_size)
        g_state['roi_manager'] = roi_manager
    
    # Soporte para formato nuevo (enabled) y formato antiguo (action)
    enabled = args.get("enabled", None)
    if enabled is not None:
        if not enabled:
            roi_manager.clear_roi()
            logging.info("ROI desactivada")
            if ws:
                await send_notification(ws, "ROI desactivada. Mostrando grid completo.", "info")
            await broadcast({
                "type": "roi_status_update",
                "payload": {"enabled": False}
            })
            return
        # Si enabled=True, continuar con set_roi
    
    action = args.get("action", "set")  # 'set', 'clear', 'get'
    
    if action == "clear" or (enabled is not None and not enabled):
        roi_manager.clear_roi()
        logging.info("ROI desactivada")
        if ws:
            await send_notification(ws, "ROI desactivada. Mostrando grid completo.", "info")
    elif action == "get":
        # Retornar información de ROI actual
        roi_info = roi_manager.get_roi_info()
        if ws:
            await send_to_websocket(ws, "roi_info", roi_info)
        return
    else:  # action == "set" o enabled == True
        x = args.get("x", 0)
        y = args.get("y", 0)
        width = args.get("width", 128)
        height = args.get("height", 128)
        
        success = roi_manager.set_roi(x, y, width, height)
        
        if success:
            roi_info = roi_manager.get_roi_info()
            reduction_msg = f" ({roi_info['reduction_ratio']:.1f}x reducción de datos)"
            logging.info(f"ROI configurada: ({x}, {y}) tamaño {width}x{height}{reduction_msg}")
            if ws:
                await send_notification(ws, f"ROI configurada: región {width}x{height} en ({x}, {y}){reduction_msg}", "info")
        else:
            error_msg = f"ROI inválida: ({x}, {y}) tamaño {width}x{height} excede el grid {roi_manager.grid_size}x{roi_manager.grid_size}"
            logging.warning(error_msg)
            if ws:
                await send_notification(ws, error_msg, "error")
            return
    
    # Enviar actualización a todos los clientes
    roi_info = roi_manager.get_roi_info()
    await broadcast({
        "type": "roi_status_update",
        "payload": roi_info
    })

async def handle_inject_energy(args):
    """Inyecta energía en el estado cuántico actual según el tipo especificado."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    motor = g_state.get('motor')
    
    if not motor:
        msg = "⚠️ No hay modelo cargado. Carga un experimento primero."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    if not motor.state or motor.state.psi is None:
        msg = "⚠️ El modelo cargado no tiene un estado válido. Intenta reiniciar la simulación."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    energy_type = args.get('type', 'primordial_soup')
    
    try:
        psi = motor.state.psi  # Shape: [batch, channels, height, width] o [channels, height, width]
        device = psi.device
        grid_size = motor.grid_size
        
        # Asegurarnos de trabajar con el tensor correcto (sin batch si existe)
        if psi.dim() == 4:  # [batch, channels, height, width]
            psi = psi[0]  # Tomar el primer batch
        # Ahora psi es [channels, height, width]
        
        channels, height, width = psi.shape[0], psi.shape[1], psi.shape[2]
        center_x, center_y = width // 2, height // 2
        
        # Crear una copia modificable del estado
        psi_new = psi.clone()
        
        if energy_type == 'primordial_soup':
            # Nebulosa de gas aleatorio alrededor del centro
            radius = min(20, grid_size // 4)
            density = 0.3
            
            for x in range(max(0, center_x - radius), min(width, center_x + radius)):
                for y in range(max(0, center_y - radius), min(height, center_y + radius)):
                    dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
                    prob = density * torch.exp(torch.tensor(-dist / (radius/2), device=device))
                    
                    if torch.rand(1, device=device).item() < prob.item():
                        # Inyectar ruido complejo aleatorio en todos los canales
                        noise = (torch.randn(channels, device=device) + 1j * torch.randn(channels, device=device)) * 0.1
                        psi_new[:, y, x] = psi_new[:, y, x] + noise
            
            logging.info(f"🧪 Sopa Primordial inyectada en el centro ({center_x}, {center_y})")
            msg = "🧪 Sopa Primordial inyectada"
            
        elif energy_type == 'dense_monolith':
            # Cubo denso y uniforme de energía
            size = min(10, grid_size // 8)
            intensity = 2.0
            
            for x in range(max(0, center_x - size), min(width, center_x + size)):
                for y in range(max(0, center_y - size), min(height, center_y + size)):
                    # Crear estado con intensidad alta
                    psi_new[:, y, x] = (torch.randn(channels, device=device) + 1j * torch.randn(channels, device=device)) * intensity * 0.1
            
            logging.info(f"⬛ Monolito Denso inyectado en el centro ({center_x}, {center_y})")
            msg = "⬛ Monolito Denso inyectado"
            
        elif energy_type == 'symmetric_seed':
            # Patrón con simetría de espejo forzada
            size = min(8, grid_size // 10)
            intensity = 1.5
            
            # Generar patrón simétrico en cuadrante superior izquierdo
            for x in range(center_x - size, center_x):
                for y in range(center_y - size, center_y):
                    # Crear estado base
                    base_state = (torch.randn(channels, device=device) + 1j * torch.randn(channels, device=device)) * intensity * 0.1
                    
                    # Reflejar en los 4 cuadrantes
                    dx, dy = center_x - x, center_y - y
                    
                    # Cuadrante 1 (original)
                    if 0 <= center_x + dx < width and 0 <= center_y + dy < height:
                        psi_new[:, center_y + dy, center_x + dx] = base_state
                    # Cuadrante 2 (reflejado en X)
                    if 0 <= center_x - dx < width and 0 <= center_y + dy < height:
                        psi_new[:, center_y + dy, center_x - dx] = base_state
                    # Cuadrante 3 (reflejado en Y)
                    if 0 <= center_x + dx < width and 0 <= center_y - dy < height:
                        psi_new[:, center_y - dy, center_x + dx] = base_state
                    # Cuadrante 4 (reflejado en ambos)
                    if 0 <= center_x - dx < width and 0 <= center_y - dy < height:
                        psi_new[:, center_y - dy, center_x - dx] = base_state
            
            logging.info(f"🔬 Semilla Simétrica inyectada en el centro ({center_x}, {center_y})")
            msg = "🔬 Semilla Simétrica inyectada"
        else:
            logging.warning(f"⚠️ Tipo de inyección desconocido: {energy_type}")
            if ws: await send_notification(ws, f"⚠️ Tipo de inyección desconocido: {energy_type}", "warning")
            return
        
        # Normalizar el estado para mantener la conservación de probabilidad
        # Normalizar por canal
        for c in range(channels):
            channel_data = psi_new[c]
            norm = torch.norm(channel_data)
            if norm > 1e-10:  # Evitar división por cero
                psi_new[c] = channel_data / norm
        
        # Restaurar la forma original si tenía batch dimension
        if motor.state.psi.dim() == 4:
            psi_new = psi_new.unsqueeze(0)
        
        # Actualizar el estado del motor
        motor.state.psi = psi_new
        
        # Reiniciar el step para indicar que el estado ha cambiado
        g_state['simulation_step'] = 0
        
        # Enviar frame actualizado si live_feed está habilitado
        live_feed_enabled = g_state.get('live_feed_enabled', True)
        if live_feed_enabled:
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
                        frame_payload_raw = {
                            "step": 0,
                            "timestamp": asyncio.get_event_loop().time(),
                            "map_data": map_data,
                            "hist_data": viz_data.get("hist_data", {}),
                            "poincare_coords": viz_data.get("poincare_coords", []),
                            "phase_attractor": viz_data.get("phase_attractor", {}),
                            "flow_data": viz_data.get("flow_data", {}),
                            "phase_hsv_data": viz_data.get("phase_hsv_data", {}),
                            "complex_3d_data": viz_data.get("complex_3d_data", {}),
                            "simulation_info": {
                                "step": 0,
                                "is_paused": g_state.get('is_paused', True),
                                "live_feed_enabled": live_feed_enabled,
                                "fps": g_state.get('current_fps', 0.0)
                            }
                        }
                        await broadcast({"type": "frame", "payload": frame_payload_raw})
            except Exception as e:
                logging.error(f"Error enviando frame después de inyección: {e}", exc_info=True)
        
        if ws: await send_notification(ws, msg, "success")
        logging.info(f"✅ Inyección de energía '{energy_type}' completada exitosamente")
        
    except Exception as e:
        logging.error(f"Error en handle_inject_energy: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al inyectar energía: {str(e)}", "error")

async def handle_set_inference_config(args):
    """Configura parámetros de inferencia (requiere recargar experimento para algunos)."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    # Parámetros que requieren recargar el experimento
    grid_size = args.get("grid_size")
    initial_state_mode = args.get("initial_state_mode")
    gamma_decay = args.get("gamma_decay")
    
    changes = []
    
    if grid_size is not None:
        old_size = global_cfg.GRID_SIZE_INFERENCE
        new_size = int(grid_size)
        global_cfg.GRID_SIZE_INFERENCE = new_size
        changes.append(f"Grid size: {old_size} → {new_size}")
        logging.info(f"Grid size de inferencia configurado a: {new_size} (requiere recargar experimento)")
        
        # Actualizar ROI manager con el nuevo tamaño
        roi_manager = g_state.get('roi_manager')
        if roi_manager:
            roi_manager.grid_size = new_size
            roi_manager.clear_roi()  # Resetear ROI al cambiar de tamaño
            logging.info(f"ROI manager actualizado con nuevo grid_size: {new_size}")
        else:
            from ..managers.roi_manager import ROIManager
            g_state['roi_manager'] = ROIManager(grid_size=new_size)
            logging.info(f"ROI manager creado con grid_size: {new_size}")
    
    if initial_state_mode is not None:
        old_mode = getattr(global_cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
        global_cfg.INITIAL_STATE_MODE_INFERENCE = str(initial_state_mode)
        changes.append(f"Inicialización: {old_mode} → {initial_state_mode}")
        logging.info(f"Modo de inicialización configurado a: {initial_state_mode} (requiere recargar experimento)")
    
    if gamma_decay is not None:
        old_gamma = getattr(global_cfg, 'GAMMA_DECAY', 0.01)
        global_cfg.GAMMA_DECAY = float(gamma_decay)
        changes.append(f"Gamma Decay: {old_gamma} → {gamma_decay}")
        logging.info(f"Gamma Decay configurado a: {gamma_decay} (requiere recargar experimento)")
    
    if changes:
        msg = f"⚠️ Configuración actualizada: {', '.join(changes)}. Recarga el experimento para aplicar los cambios."
        if ws:
            await send_notification(ws, msg, "warning")
        
        await broadcast({
            "type": "inference_config_update",
            "payload": {
                "grid_size": global_cfg.GRID_SIZE_INFERENCE,
                "initial_state_mode": global_cfg.INITIAL_STATE_MODE_INFERENCE,
                "gamma_decay": global_cfg.GAMMA_DECAY
            }
        })
    else:
        if ws:
            await send_notification(ws, "No se especificaron cambios en la configuración.", "info")

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
        from ..managers.history_manager import HISTORY_DIR
        import json
        from pathlib import Path
        
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
        from ..managers.history_manager import HISTORY_DIR
        import json
        from pathlib import Path
        
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

async def handle_capture_snapshot(args):
    """Captura un snapshot manual del estado actual de la simulación."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    try:
        motor = g_state.get('motor')
        if not motor:
            msg = "⚠️ No hay un modelo cargado. Primero debes cargar un experimento."
            logging.warning(msg)
            if ws:
                await send_notification(ws, msg, "warning")
            return
        
        if motor.state.psi is None:
            msg = "⚠️ El motor no tiene un estado válido para capturar."
            logging.warning(msg)
            if ws:
                await send_notification(ws, msg, "warning")
            return
        
        # Inicializar lista de snapshots si no existe
        if 'snapshots' not in g_state:
            g_state['snapshots'] = []
        
        current_step = g_state.get('simulation_step', 0)
        
        # Capturar snapshot
        try:
            psi_tensor = motor.state.psi
            snapshot = psi_tensor.detach().cpu().clone() if hasattr(psi_tensor, 'detach') else psi_tensor.cpu().clone()
            
            g_state['snapshots'].append({
                'psi': snapshot,
                'step': current_step,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            # Limitar número de snapshots almacenados
            max_snapshots = g_state.get('max_snapshots', 500)
            if len(g_state['snapshots']) > max_snapshots:
                g_state['snapshots'] = g_state['snapshots'][-max_snapshots:]
            
            n_snapshots = len(g_state['snapshots'])
            msg = f"📸 Snapshot capturado manualmente (paso {current_step}). Total: {n_snapshots} snapshots."
            logging.info(msg)
            if ws:
                await send_notification(ws, msg, "success")
                # Enviar actualización del número de snapshots
                await send_to_websocket(ws, "snapshot_count", {
                    "count": n_snapshots,
                    "step": current_step
                })
        except Exception as e:
            error_msg = f"Error al capturar snapshot: {str(e)}"
            logging.error(error_msg, exc_info=True)
            if ws:
                await send_notification(ws, error_msg, "error")
        
    except Exception as e:
        logging.error(f"Error en handle_capture_snapshot: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"Error al capturar snapshot: {str(e)}", "error")

# Diccionario central para mapear comandos a funciones handler
# Se define aquí después de todas las funciones handler
HANDLERS = {
    "experiment": {
        "create": handle_create_experiment, 
        "continue": handle_continue_experiment, 
        "stop": handle_stop_training,
        "delete": handle_delete_experiment,
        "list_checkpoints": handle_list_checkpoints,
        "delete_checkpoint": handle_delete_checkpoint,
        "cleanup_checkpoints": handle_cleanup_checkpoints
    },
    "simulation": {
        "set_viz": handle_set_viz,
        "update_visualization": handle_update_visualization,  # Nuevo: actualización manual
        "set_speed": handle_set_simulation_speed,
        "set_fps": handle_set_fps,
        "set_frame_skip": handle_set_frame_skip,
        "set_live_feed": handle_set_live_feed,
        "set_steps_interval": handle_set_steps_interval,
        "set_compression": handle_set_compression,
        "set_downsample": handle_set_downsample,
        "set_roi": handle_set_roi,
        "set_snapshot_interval": handle_set_snapshot_interval,
        "enable_snapshots": handle_enable_snapshots,
        "capture_snapshot": handle_capture_snapshot,
        "enable_history": handle_enable_history,
        "save_history": handle_save_history,
        "clear_history": handle_clear_history,
        "list_history_files": handle_list_history_files,
        "load_history_file": handle_load_history_file
    },
    "analysis": {
        "universe_atlas": handle_analyze_universe_atlas,
        "cell_chemistry": handle_analyze_cell_chemistry,
        "cancel": handle_cancel_analysis,
        "clear_snapshots": handle_clear_snapshots
    },
    "inference": {
        "play": handle_play, 
        "pause": handle_pause, 
        "load": handle_load_experiment,  # También acepta "load" además de "load_experiment"
        "load_experiment": handle_load_experiment, 
        "unload": handle_unload_model,  # Nuevo: descargar modelo
        "switch_engine": handle_switch_engine,  # Nuevo: cambiar entre motor nativo y Python
        "reset": handle_reset,
        "set_config": handle_set_inference_config,
        "inject_energy": handle_inject_energy
    },
    "server": {
        "shutdown": handle_shutdown  # Nuevo: apagar servidor desde UI
    },
    "system": {
        "refresh_experiments": handle_refresh_experiments
    }
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
    app.router.add_get("/ws", websocket_handler)
    
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
