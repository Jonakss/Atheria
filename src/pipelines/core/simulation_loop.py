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
                        # CRÍTICO: Si live_feed está desactivado, forzar comportamiento de fullspeed
                        # a menos que steps_interval sea muy grande (>1000) para actualizaciones lentas
                        effective_steps_interval = steps_interval
                        if not live_feed_enabled and steps_interval != -1 and steps_interval < 1000:
                            effective_steps_interval = -1
                        
                        # Determinar cuántos pasos ejecutar en esta iteración
                        # Si effective_steps_interval es 0 (modo manual), ejecutar un número razonable de pasos
                        # Si effective_steps_interval es -1 (modo fullspeed), ejecutar un número grande de pasos
                        # De lo contrario, ejecutar hasta effective_steps_interval pasos
                        if effective_steps_interval == 0:
                            steps_to_execute = 100  # Modo manual: ejecutar múltiples pasos para velocidad
                        elif effective_steps_interval == -1:
                            steps_to_execute = 1000  # Modo fullspeed: ejecutar muchos pasos para máxima velocidad
                        else:
                            steps_to_execute = effective_steps_interval
                        
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
                                try:
                                    # Timeout de 5s para evitar bloqueos infinitos en motor nativo
                                    # Pasamos el paso objetivo (current + idx + 1) para caching correcto
                                    target_step = current_step + step_idx + 1
                                    await asyncio.wait_for(
                                        asyncio.get_event_loop().run_in_executor(None, lambda: motor.evolve_internal_state(step=target_step)),
                                        timeout=5.0
                                    )
                                except asyncio.TimeoutError:
                                    logging.error("❌ Timeout crítico en motor.evolve_internal_state (5s). El motor nativo parece bloqueado.")
                                    g_state['is_paused'] = True
                                    await broadcast({"type": "error", "payload": {"message": "Motor nativo bloqueado (timeout). Pausando simulación."}})
                                    break
                                except Exception as e:
                                    logging.error(f"❌ Error en motor.evolve_internal_state: {e}")
                                    g_state['is_paused'] = True
                                    break

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
                        # Para live_feed ON: mostrar frames/segundo (o pasos/segundo si steps_interval > 1)
                        
                        # Lógica unificada de FPS:
                        # Siempre calcular y actualizar current_fps basado en el rendimiento real
                        if 'current_fps' not in g_state or 'fps_samples' not in g_state:
                            g_state['current_fps'] = min(steps_per_second, 10000.0)
                            g_state['fps_samples'] = [min(steps_per_second, 10000.0)]
                        else:
                            # Promediar con últimos valores para suavizar
                            fps_value = min(steps_per_second, 10000.0)
                            g_state['fps_samples'].append(fps_value)
                            # Mantener ventana de muestras (ej: 10)
                            if len(g_state['fps_samples']) > 10:
                                g_state['fps_samples'].pop(0)
                        
                        if 'fps_samples' in g_state and len(g_state['fps_samples']) > 0:
                            g_state['current_fps'] = sum(g_state['fps_samples']) / len(g_state['fps_samples'])
                            
                            # DEBUG: Log FPS calculation every 100 steps
                            if updated_step % 100 == 0:
                                logging.info(f"📊 FPS calculado: {g_state['current_fps']:.1f} (live_feed={live_feed_enabled}, steps_interval={steps_interval})")
                        
                        # Actualizar contador para frames (solo si no es modo manual)
                        steps_interval_counter = g_state.get('steps_interval_counter', 0)
                        steps_interval_counter += steps_to_execute
                        g_state['steps_interval_counter'] = steps_interval_counter
                        
                        # Enviar frame cada X pasos configurados
                        # Modo manual (steps_interval = 0): NO enviar frames automáticamente
                        # Modo fullspeed (steps_interval = -1): NO enviar frames NUNCA
                        # También enviar frame si nunca se ha enviado uno (last_frame_sent_step == -1) EXCEPTO en fullspeed
                        if effective_steps_interval == -1:
                            # Modo fullspeed: NUNCA enviar frames
                            should_send_frame = False
                        elif effective_steps_interval == 0:
                            # Modo manual: NO enviar frames automáticamente
                            # Solo enviar el primer frame si nunca se ha enviado uno
                            should_send_frame = (g_state['last_frame_sent_step'] == -1)
                        else:
                            # Modo automático: enviar frame cada N pasos
                            should_send_frame = (steps_interval_counter >= effective_steps_interval) or (g_state['last_frame_sent_step'] == -1)
                        
                        if should_send_frame:
                            # Resetear contador
                            g_state['steps_interval_counter'] = 0
                            
                            # Detectar época periódicamente (cada 50 pasos para no saturar)
                            # Detectar época y calcular métricas periódicamente (cada 50 pasos)
                            epoch_detector = g_state.get('epoch_detector')
                            # Calcular métricas si hay detector O si estamos en inferencia (para Harlow Limit)
                            should_calculate_metrics = (updated_step % 50 == 0)
                            
                            if should_calculate_metrics:
                                try:
                                    # OPTIMIZACIÓN: Para motor nativo, usar get_dense_state() si está disponible
                                    motor = g_state['motor']
                                    motor_is_native = g_state.get('motor_is_native', False)
                                    if motor_is_native and hasattr(motor, 'get_dense_state'):
                                        psi_tensor = motor.get_dense_state(check_pause_callback=lambda: g_state.get('is_paused', True))
                                    else:
                                        psi_tensor = motor.state.psi if hasattr(motor, 'state') and motor.state else None
                                    
                                    if psi_tensor is not None:
                                        metrics = {}
                                        if epoch_detector:
                                            # Analizar estado y determinar época
                                            try:
                                                metrics = epoch_detector.analyze_state(psi_tensor)
                                                epoch = epoch_detector.determine_epoch(metrics)
                                                g_state['current_epoch'] = epoch
                                            except RuntimeError as e:
                                                if "size of tensor a" in str(e) and "match the size of tensor b" in str(e):
                                                    # Mismatch de dimensiones (ej: LatticeEngine vs UNet)
                                                    # Desactivar detector para esta sesión para evitar spam
                                                    logging.warning(f"⚠️ Desactivando EpochDetector por incompatibilidad de dimensiones: {e}")
                                                    g_state['epoch_detector'] = None
                                                else:
                                                    raise e
                                        
                                        # Calcular métricas adicionales para Harlow Limit (siempre)
                                        from ...physics.metrics import calculate_fidelity, calculate_entanglement_entropy
                                        
                                        # Necesitamos estado inicial para fidelidad
                                        # Si no tenemos estado inicial guardado, usar el actual como referencia (F=1)
                                        if 'initial_state_ref' not in g_state:
                                            g_state['initial_state_ref'] = psi_tensor.clone().detach()
                                            
                                        fidelity = calculate_fidelity(g_state['initial_state_ref'], psi_tensor)
                                        entropy = calculate_entanglement_entropy(psi_tensor)
                                        
                                        # Actualizar epoch_metrics (merge con existentes o defaults)
                                        current_metrics = g_state.get('epoch_metrics', {})
                                        new_metrics = {
                                            'energy': float(metrics.get('energy', current_metrics.get('energy', 0))),
                                            'clustering': float(metrics.get('clustering', current_metrics.get('clustering', 0))),
                                            'symmetry': float(metrics.get('symmetry', current_metrics.get('symmetry', 0))),
                                            'fidelity': float(fidelity),
                                            'entropy': float(entropy)
                                        }
                                        g_state['epoch_metrics'] = new_metrics
                                except Exception as e:
                                    logging.warning(f"Error calculando métricas/época: {e}")
                            
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
                            # OPTIMIZACIÓN CRÍTICA: Usar lazy conversion para motor nativo
                            # Solo convertir estado denso cuando se necesita visualizar
                            motor = g_state['motor']
                            motor_is_native = g_state.get('motor_is_native', False)
                            
                            # Verificar si podemos usar visualización nativa rápida (C++)
                            # Tipos soportados por C++: density, phase, energy
                            native_viz_supported = viz_type in ['density', 'phase', 'energy']
                            
                            # Determinar si necesitamos el estado denso (psi) obligatoriamente
                            need_dense_state = False
                            
                            # 1. Epoch detector lo necesita cada 50 pasos
                            if epoch_detector and updated_step % 50 == 0:
                                need_dense_state = True
                                
                            # 2. Si el tipo de visualización NO es soportado nativamente, necesitamos psi
                            if not native_viz_supported:
                                need_dense_state = True
                                
                            # 3. Si queremos guardar historial completo (con psi) cada N pasos
                            # Por ahora, priorizamos rendimiento y aceptamos historial "solo visual" en nativo
                            
                            psi = None
                            viz_data = None
                            
                            # FAST PATH: Visualización Nativa (C++)
                            # Evita la costosa conversión sparse->dense en Python
                            if motor_is_native and hasattr(motor, 'get_visualization_data') and not need_dense_state:
                                try:
                                    # Llamada directa a C++ (retorna tensor [H, W])
                                    viz_tensor = await asyncio.get_event_loop().run_in_executor(
                                        None,
                                        lambda: motor.get_visualization_data(viz_type)
                                    )
                                    
                                    if viz_tensor is not None:
                                        # Convertir a lista para JSON (rápido en CPU)
                                        if viz_tensor.is_cuda:
                                            viz_tensor = viz_tensor.cpu()
                                        
                                        map_data = viz_tensor.tolist()
                                        
                                        # Construir objeto viz_data mínimo
                                        viz_data = {
                                            "map_data": map_data,
                                            "hist_data": {}, # Histograma no disponible en fast path
                                            "poincare_coords": [],
                                            "phase_attractor": None,
                                            "flow_data": None
                                        }
                                        
                                        # Si es 'phase', normalizar si es necesario (el frontend espera radianes o [0,1])
                                        # C++ devuelve radianes [-pi, pi] o valor crudo
                                except Exception as e:
                                    logging.error(f"❌ Error en visualización nativa rápida: {e}. Cayendo a fallback.")
                                    need_dense_state = True # Fallback a camino lento
                            
                            # SLOW PATH: Obtener estado denso y procesar en Python
                            # Se ejecuta si no es nativo, o si necesitamos psi (epoch, viz avanzada, error en fast path)
                            if psi is None and (not motor_is_native or need_dense_state):
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
                                    # CRÍTICO: Usar timeout para evitar bloqueo indefinido en grids grandes
                                    try:
                                        psi = await asyncio.wait_for(
                                            asyncio.get_event_loop().run_in_executor(
                                                None, 
                                                lambda: motor.get_dense_state(roi=roi, check_pause_callback=check_pause)
                                            ),
                                            timeout=10.0  # Timeout de 10s para grids grandes
                                        )
                                    except asyncio.TimeoutError:
                                        logging.error(f"❌ Timeout crítico en get_dense_state (10s) para grid {g_state.get('inference_grid_size')}. Saltando frame.")
                                        g_state['is_paused'] = True
                                        await broadcast({"type": "error", "payload": {"message": "get_dense_state bloqueado (timeout 10s). Pausando simulación. Intenta reducir grid size o usar ROI."}})
                                        await asyncio.sleep(0.1)
                                        continue
                                    except Exception as e:
                                        logging.error(f"❌ Error en get_dense_state: {e}")
                                        await asyncio.sleep(0.1)
                                        continue
                                        
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
                            
                            # Verificar que tengamos datos para visualizar (psi o viz_data)
                            if psi is None and viz_data is None:
                                logging.error("❌ Estado psi y viz_data son None. Saltando frame.")
                                continue
                            
                            # DEBUG: Verificar que psi tenga datos válidos (solo si existe)
                            if psi is not None and isinstance(psi, torch.Tensor):
                                if psi.numel() == 0:
                                    logging.error("❌ psi tiene 0 elementos. Saltando frame.")
                                    continue
                                psi_abs_max = psi.abs().max().item()
                                psi_mean = psi.abs().mean().item()
                                if psi_abs_max < 1e-10:
                                    logging.error(f"❌ CRÍTICO: psi tiene valores muy pequeños (max abs={psi_abs_max:.6e}, mean={psi_mean:.6e}). El motor está vacío o no inicializado correctamente.")
                                    logging.warning(f"⚠️ CRÍTICO: psi tiene valores muy pequeños (max abs={psi_abs_max:.6e}, mean={psi_mean:.6e}). El motor está vacío o no inicializado correctamente.")
                                    # NOTA: No inyectamos partículas artificialmente. Deben emerger del vacío o ser sembradas inicialmente.

                            # Calcular visualización si no la tenemos ya (Fast Path la provee)
                            if viz_data is None:
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
                                                "start_step": g_state.get('start_step', 0), # Deprecated, use initial_step
                                                "initial_step": g_state.get('initial_step', 0),
                                                "session_steps": updated_step - g_state.get('initial_step', 0),
                                                "total_steps": updated_step,
                                                "checkpoint_step": g_state.get('checkpoint_step', 0),
                                                "checkpoint_episode": g_state.get('checkpoint_episode', 0),
                                                "is_paused": False,
                                                "live_feed_enabled": live_feed_enabled, # Use live_feed_enabled here
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
                                        
                                        # 1. Aplicar ROI primero (reduce el tamaño de los datos)
                                        roi_manager = g_state.get('roi_manager')
                                        if roi_manager and roi_manager.roi_enabled:
                                            from ...managers.roi_manager import apply_roi_to_payload
                                            frame_payload_roi = apply_roi_to_payload(frame_payload_raw, roi_manager)
                                        else:
                                            frame_payload_roi = frame_payload_raw

                                        if compression_enabled or downsample_factor > 1:
                                            frame_payload = await optimize_frame_payload(
                                                frame_payload_roi,
                                                enable_compression=compression_enabled,
                                                downsample_factor=downsample_factor,
                                                viz_type=g_state.get('viz_type', 'density')
                                            )
                                        else:
                                            frame_payload = frame_payload_roi
                                        
                                        await broadcast({"type": "simulation_frame", "payload": frame_payload})
                                        frame_count += 1
                                        g_state['last_frame_sent_step'] = updated_step  # Marcar que se envió un frame
                                        
                                        # Agregar al historial (buffer circular)
                                        # Usamos frame_payload_raw para tener los datos sin comprimir/serializar
                                        if 'simulation_history' in g_state and g_state.get('history_enabled', True):
                                            # Crear payload para historial que incluye el estado psi (si existe)
                                            # Esto permite restaurar la simulación exacta, no solo visualizar
                                            history_payload = frame_payload_raw.copy()
                                            
                                            # Si tenemos psi disponible (y no es None), guardarlo en CPU
                                            # psi variable is available in this scope (from line 332/318)
                                            if psi is not None and isinstance(psi, torch.Tensor):
                                                # Detach y mover a CPU para no ocupar VRAM
                                                history_payload['psi'] = psi.detach().cpu()
                                            
                                            # Solo guardar cada N frames para no saturar memoria si es muy rápido
                                            # Pero para rewind suave idealmente queremos todos o casi todos
                                            # El buffer es limitado (1000), así que se sobrescribirá
                                            g_state['simulation_history'].add_frame(history_payload)
                                else:
                                    logging.warning(f"⚠️ map_data está vacío o inválido en step {updated_step}. Saltando frame.")
                                    # No continue here, just skip sending frame
                            else:
                                logging.warning(f"⚠️ viz_data es None o inválido en step {updated_step}. Saltando frame.")
                                # No continue here, just skip sending frame
                        else:
                            logging.debug(f"Skipping frame generation for step {updated_step} (should_send_frame is False)")
                            
                        # THROTTLE: Solo enviar actualización de estado cada STATE_UPDATE_INTERVAL segundos
                        # para evitar saturar el WebSocket con demasiados mensajes
                        # IMPORTANTE: NO enviar state_update en modo fullspeed (steps_interval == -1)
                        current_time = time.time()
                        time_since_last_update = current_time - last_state_update_time
                        
                        # Solo enviar actualización si NO estamos en modo fullspeed
                        if steps_interval != -1 and time_since_last_update >= STATE_UPDATE_INTERVAL:
                            # Enviar actualización de estado (sin datos de visualización pesados)
                            # Esto permite que el frontend muestre el progreso aunque no haya visualización
                            state_update = {
                                "step": updated_step,
                                "timestamp": asyncio.get_event_loop().time(),
                                "simulation_info": {
                                    "step": updated_step,
                                    "start_step": g_state.get('start_step', 0),
                                    "initial_step": g_state.get('initial_step', 0),
                                    "session_steps": updated_step - g_state.get('initial_step', 0),
                                    "total_steps": updated_step,
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
                        # IMPORTANTE: NO enviar logs en modo fullspeed (steps_interval == -1)
                        if steps_interval != -1 and updated_step % 100 == 0:
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

