"""Bucle principal de simulación que evoluciona el estado y difunde datos de visualización."""
import asyncio
import logging
import time

from ...server.server_state import (
    g_state, 
    broadcast, 
    optimize_frame_payload, 
    get_payload_size, 
    apply_roi_to_payload
)
from ..viz import get_visualization_data
from .helpers import calculate_adaptive_downsample, calculate_adaptive_roi
from .status_helpers import build_inference_status_payload

logger = logging.getLogger(__name__)


async def simulation_loop():
    """Bucle principal que evoluciona el estado y difunde los datos de visualización."""
    logging.debug("Iniciando bucle de simulación.")
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
                
                # OPTIMIZACIÓN CRÍTICA: Ejecutar múltiples pasos por frame según steps_interval
                # Esto permite acelerar la simulación incluso con live_feed activado
                # Si live_feed está desactivado, steps_interval controla la frecuencia de actualización de estado
                # Si live_feed está activado, steps_interval controla cuántos pasos se ejecutan por cada frame visualizado
                
                # Obtener intervalo de pasos configurado (por defecto 10 si live_feed OFF, 1 si ON para mantener comportamiento anterior por defecto)
                default_interval = 10 if not live_feed_enabled else 1
                steps_interval = g_state.get('steps_interval', default_interval)
                
                # Forzar al menos 1 paso si no es modo manual/fullspeed
                if steps_interval > 0:
                    steps_interval = max(1, steps_interval)
                
                if 'steps_interval_counter' not in g_state:
                    g_state['steps_interval_counter'] = 0
                if 'last_frame_sent_step' not in g_state:
                    g_state['last_frame_sent_step'] = -1  # Para forzar primer frame

                # Lógica unificada para ejecución de pasos (con o sin live feed)
                # Si steps_interval > 1, ejecutamos múltiples pasos en un bucle rápido
                if True: # Bloque unificado (reemplaza el if not live_feed_enabled)
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
                                # Offload evolution to thread pool to avoid blocking event loop
                                await asyncio.get_event_loop().run_in_executor(None, motor.evolve_internal_state)
                            updated_step = current_step + step_idx + 1
                            g_state['simulation_step'] = updated_step
                            steps_executed_this_iteration += 1
                            
                            # CRÍTICO: Yield al event loop periódicamente para permitir procesar comandos WebSocket
                            # Esto previene que el simulation_loop bloquee el event loop y permita respuesta rápida a comandos
                            # Yield cada paso para motor nativo (más frecuente), cada 10 para motor Python
                            if (step_idx + 1) % (1 if motor_is_native else 10) == 0:
                                await asyncio.sleep(0)  # Yield al event loop para permitir otros tasks
                            
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
                                
                                # CRÍTICO: Verificar pausa ANTES de conversión costosa
                                if g_state.get('is_paused', True):
                                    logging.info("⏸️ Pausa detectada antes de get_dense_state. Saltando frame.")
                                    await asyncio.sleep(0.1)
                                    continue
                                
                                # Obtener estado denso (solo convierte si es necesario)
                                # Offload to thread pool
                                psi = await asyncio.get_event_loop().run_in_executor(
                                    None, 
                                    lambda: motor.get_dense_state(roi=roi, check_pause_callback=check_pause)
                                )
                                # CRÍTICO: Yield al event loop después de conversión bloqueante (puede tardar en grids grandes)
                                await asyncio.sleep(0)  # Permitir procesar comandos WebSocket
                                
                                # CRÍTICO: Verificar pausa DESPUÉS de conversión costosa
                                if g_state.get('is_paused', True):
                                    logging.info("⏸️ Pausa detectada después de get_dense_state. Saltando frame.")
                                    await asyncio.sleep(0.1)
                                    continue
                            else:
                                # Motor Python: acceder directamente (ya es denso)
                                psi = motor.state.psi if hasattr(motor, 'state') and motor.state else None
                            
                            # Verificar que psi no sea None
                            if psi is None:
                                logging.error("❌ Estado psi es None. Saltando frame.")
                                continue
                            
                            # DEBUG: Verificar que psi tenga datos válidos
                            if isinstance(psi, torch.Tensor):
                                if psi.numel() == 0:
                                    logging.error("❌ psi tiene 0 elementos. Saltando frame.")
                                    continue
                                psi_abs_max = psi.abs().max().item()
                                psi_mean = psi.abs().mean().item()
                                if psi_abs_max < 1e-10:
                                    logging.error(f"❌ CRÍTICO: psi tiene valores muy pequeños (max abs={psi_abs_max:.6e}, mean={psi_mean:.6e}). El motor está vacío o no inicializado correctamente.")
                                    logging.warning(f"⚠️ CRÍTICO: psi tiene valores muy pequeños (max abs={psi_abs_max:.6e}, mean={psi_mean:.6e}). El motor está vacío o no inicializado correctamente.")
                                    # NOTA: No inyectamos partículas artificialmente. Deben emerger del vacío o ser sembradas inicialmente.

                                # 3. Generar frame de visualización (si corresponde)
                                current_time = time.time()
                                frame_data = None
                                
                                # Decidir si generar frame
                                should_generate_frame = (
                                    live_feed_enabled and 
                                    (current_time - last_frame_time >= target_frame_time)
                                )
                                
                                if should_generate_frame:
                                    try:
                                        # Generar datos de visualización
                                        viz_start = time.time()
                                        frame_data = await self.viz_pipeline.generate_frame(
                                            state, 
                                            step_count,
                                            viz_type=self.viz_pipeline.current_viz_type
                                        )
                                        viz_time = time.time() - viz_start
                                        
                                        if frame_data:
                                            # Añadir métricas de rendimiento
                                            frame_data['simulation_info'] = {
                                                'step': step_count,
                                                'fps': 1.0 / (current_time - last_frame_time) if last_frame_time > 0 else 0,
                                                'sim_time': step_time,
                                                'viz_time': viz_time,
                                                'live_feed_enabled': live_feed_enabled,
                                                'inference_grid_size': state.get('grid_size', 0), # Tamaño real
                                                'training_grid_size': state.get('training_grid_size', 0)
                                            }
                                            
                                            # Log para debug de "trancado"
                                            logging.info(f"Frame generado - Step: {step_count}, Viz Time: {viz_time:.4f}s, Map Data: {'map_data' in frame_data}")
                                            
                                            # Enviar al cliente
                                            await broadcast(frame_data, binary=True)
                                            last_frame_time = current_time
                                        else:
                                            logging.warning(f"Frame generado vacío en paso {step_count}")
                                            
                                    except Exception as e:
                                        logging.error(f"Error generando/enviando frame: {e}")
                                        import traceback
                                        traceback.print_exc()
                            else:
                                logging.error(f"⚠️ psi no es un torch.Tensor: {type(psi)}")
                                continue
                            
                            # OPTIMIZACIÓN PARA GRIDS GRANDES: Downsampling adaptativo y ROI automático
                            inference_grid_size = g_state.get('inference_grid_size', 256)
                            adaptive_downsample = calculate_adaptive_downsample(inference_grid_size)
                            
                            # Aplicar ROI automático para grids muy grandes (>512)
                            if inference_grid_size > 512 and motor_is_native and hasattr(motor, 'get_dense_state'):
                                adaptive_roi = calculate_adaptive_roi(inference_grid_size)
                                roi_manager = g_state.get('roi_manager')
                                if adaptive_roi and roi_manager and not roi_manager.roi_enabled:
                                    # Activar ROI automático solo si no está ya habilitado manualmente
                                    roi_manager.set_roi(*adaptive_roi)
                                    roi_manager.roi_enabled = True
                                    logging.info(f"🔍 ROI automático activado para grid grande ({inference_grid_size}x{inference_grid_size}): {adaptive_roi}")
                            
                            if adaptive_downsample > 1:
                                logging.debug(f"🔄 Downsampling adaptativo activado: factor={adaptive_downsample} para grid {inference_grid_size}x{inference_grid_size}")
                            
                            delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                            viz_data = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: get_visualization_data(
                                    psi, 
                                    viz_type,
                                    delta_psi=delta_psi,
                                    motor=motor,
                                    downsample_factor=adaptive_downsample
                                )
                            )
                            
                            # OPTIMIZACIÓN: Reutilizar coordenadas de Poincaré del frame anterior si no se recalcula
                            if should_calc_poincare and not calc_poincare_this_frame and 'last_poincare_coords' in g_state:
                                viz_data['poincare_coords'] = g_state['last_poincare_coords']
                            elif should_calc_poincare and calc_poincare_this_frame and 'poincare_coords' in viz_data:
                                g_state['last_poincare_coords'] = viz_data['poincare_coords']
                            
                            if viz_data and isinstance(viz_data, dict):
                                map_data = viz_data.get("map_data", [])
                                if map_data and len(map_data) > 0:
                                    # DEBUG: Log cuando se envía frame con map_data
                                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                                        map_data_min = min(min(row) if row else 0 for row in map_data) if map_data else 0
                                        map_data_max = max(max(row) if row else 0 for row in map_data) if map_data else 0
                                        logging.debug(f"📤 Enviando simulation_frame - step: {updated_step}, map_data shape: [{len(map_data)}, {len(map_data[0]) if map_data else 0}], map_data range: [{map_data_min:.6f}, {map_data_max:.6f}]")
                                    
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
                        logging.error(f"Error evolucionando estado: {e}", exc_info=True)
                    
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
                        # PERO si steps_interval > 1, queremos ir rápido entre frames
                        if steps_interval > 1:
                             # Si estamos saltando pasos, permitir ir rápido
                             await asyncio.sleep(0)
                        else:
                             sleep_time = max(0.016, ideal_sleep)  # Mínimo 16ms cuando hay live feed normal
                             await asyncio.sleep(sleep_time)
                    
                    # Si no se debe enviar frame, continuar al siguiente ciclo del while
                    if not should_send_frame:
                        continue
                
                # AQUI CONTINUA LA LÓGICA DE VISUALIZACIÓN (solo si should_send_frame es True)
                # El bloque 'else' original (líneas 507-511) se elimina porque ya ejecutamos los pasos arriba
                
                # Validar que el motor tenga un estado válido
                # CRÍTICO: Para motor nativo, el estado está en C++, no en motor.state.psi
                motor_is_native = g_state.get('motor_is_native', False)
                if not motor_is_native and g_state['motor'].state.psi is None:
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
                        
                        # CRÍTICO: Verificar pausa ANTES de conversión costosa
                        if g_state.get('is_paused', True):
                            logging.info("⏸️ Pausa detectada antes de get_dense_state. Saltando frame.")
                            await asyncio.sleep(0.1)
                            continue
                        
                        # Obtener estado denso (solo convierte si es necesario)
                        # Offload to thread pool
                        psi = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: motor.get_dense_state(roi=roi, check_pause_callback=check_pause)
                        )
                        # CRÍTICO: Yield al event loop después de conversión bloqueante
                        await asyncio.sleep(0)  # Permitir procesar comandos WebSocket
                        
                        # CRÍTICO: Verificar pausa DESPUÉS de conversión costosa
                        if g_state.get('is_paused', True):
                            logging.info("⏸️ Pausa detectada después de get_dense_state. Saltando frame.")
                            await asyncio.sleep(0.1)
                            continue
                    else:
                        # Motor Python: acceder directamente (ya es denso)
                        psi = motor.state.psi if hasattr(motor, 'state') and motor.state else None
                    
                    # Verificar que psi no sea None
                    if psi is None:
                        logging.warning("Estado psi es None. Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # CRÍTICO: Verificar pausa antes de cálculo de visualización
                    if g_state.get('is_paused', True):
                        logging.info("⏸️ Pausa detectada antes de get_visualization_data. Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Optimización: Usar inference_mode para mejor rendimiento GPU
                    # Obtener delta_psi si está disponible para visualizaciones de flujo
                    delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                    # Offload viz calculation to thread pool
                    viz_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: get_visualization_data(
                            psi, 
                            g_state.get('viz_type', 'density'),
                            delta_psi=delta_psi,
                            motor=motor
                        )
                    )
                    # CRÍTICO: Yield al event loop después de cálculo de visualización (puede ser bloqueante)
                    await asyncio.sleep(0)  # Permitir procesar comandos WebSocket
                    
                    # Validar que viz_data tenga los campos necesarios
                    if not viz_data or not isinstance(viz_data, dict):
                        logging.warning(f"⚠️ get_visualization_data retornó datos inválidos (tipo: {type(viz_data)}). Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Validar que map_data no esté vacío
                    # CRÍTICO: map_data ahora puede ser numpy array o lista
                    map_data = viz_data.get("map_data", [])
                    if map_data is None:
                        logging.warning(f"⚠️ map_data es None en step {g_state.get('simulation_step', 0)}. Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Validar tamaño según tipo
                    import numpy as np
                    if isinstance(map_data, np.ndarray):
                        if map_data.size == 0:
                            logging.warning(f"⚠️ map_data (numpy array) está vacío en step {g_state.get('simulation_step', 0)}. Saltando frame.")
                            await asyncio.sleep(0.1)
                            continue
                    elif isinstance(map_data, list):
                        if len(map_data) == 0:
                            logging.warning(f"⚠️ map_data (lista) está vacío en step {g_state.get('simulation_step', 0)}. Saltando frame.")
                            await asyncio.sleep(0.1)
                            continue
                    else:
                        logging.warning(f"⚠️ map_data tiene tipo inesperado: {type(map_data)} en step {g_state.get('simulation_step', 0)}. Saltando frame.")
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
                        from ...managers.roi_manager import apply_roi_to_payload
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
                            import torch
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
        status_payload = build_inference_status_payload("paused")
        await broadcast({"type": "inference_status_update", "payload": status_payload})
        await asyncio.sleep(2)

