# src/pipeline_server.py
import asyncio
import json
import logging
import os
import uuid
from aiohttp import web
from pathlib import Path

# Asumimos la existencia y correcto funcionamiento de estos módulos locales
from . import config as global_cfg
from .server_state import g_state, broadcast, send_notification, send_to_websocket
from .utils import get_experiment_list, load_experiment_config, get_latest_checkpoint
from .server_handlers import create_experiment_handler
from .pipeline_viz import get_visualization_data
from .model_loader import load_model
from .qca_engine import Aetheria_Motor, QuantumState
from .analysis import analyze_universe_atlas, analyze_cell_chemistry, calculate_phase_map_metrics

# Configuración de logging para ver claramente lo que pasa en el servidor
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# --- WEBSOCKET HANDLER ---
async def websocket_handler(request):
    """Maneja las conexiones WebSocket entrantes."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    ws_id = str(uuid.uuid4())
    g_state['websockets'][ws_id] = ws
    logging.info(f"Nueva conexión WebSocket: {ws_id}")
    
    # Enviar estado inicial al cliente
    experiments = get_experiment_list()
    initial_state = {
        "type": "initial_state",
        "payload": {
            "experiments": experiments,
            "training_status": "running" if g_state.get('training_process') else "idle",
            "inference_status": "running" if not g_state.get('is_paused', True) else "paused"
        }
    }
    await ws.send_json(initial_state)
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    scope = data.get('scope')
                    command = data.get('command')
                    args = data.get('args', {})
                    args['ws_id'] = ws_id
                    
                    logging.info(f"Comando recibido: {scope}.{command} de [{ws_id}]")
                    
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
                    await send_notification(ws, f"Error al ejecutar comando: {str(e)}", "error")
            elif msg.type == web.WSMsgType.ERROR:
                logging.error(f"Error en WebSocket: {ws.exception()}")
    finally:
        # Limpiar la conexión
        if ws_id in g_state['websockets']:
            del g_state['websockets'][ws_id]
        logging.info(f"Conexión WebSocket cerrada: {ws_id}")
    
    return ws

# Reemplaza esta función en tu src/pipeline_server.py
async def simulation_loop():
    """Bucle principal que evoluciona el estado y difunde los datos de visualización."""
    logging.info("Iniciando bucle de simulación (actualmente en pausa).")
    while True:
        try:
            is_paused = g_state.get('is_paused', True)
            motor = g_state.get('motor')
            
            # Log de diagnóstico ocasional (cada 5 segundos aproximadamente)
            if motor is None and not is_paused:
                # Solo loguear ocasionalmente para no saturar
                import time
                if not hasattr(simulation_loop, '_last_warning_time'):
                    simulation_loop._last_warning_time = 0
                current_time = time.time()
                if current_time - simulation_loop._last_warning_time > 5:
                    logging.warning("Simulación en ejecución pero sin motor cargado. Carga un modelo para ver datos.")
                    simulation_loop._last_warning_time = current_time
            
            if not is_paused and motor:
                current_step = g_state.get('simulation_step', 0)
                try:
                    # Evolucionar el estado (siempre, para mantener la física correcta)
                    g_state['motor'].evolve_internal_state()
                    g_state['simulation_step'] = current_step + 1
                    
                    # OPTIMIZACIÓN: Solo calcular y enviar visualizaciones si live_feed está activo
                    # Esto libera recursos cuando el usuario no está visualizando
                    live_feed_enabled = g_state.get('live_feed_enabled', True)
                    
                    # Si live_feed está desactivado, solo evolucionar el estado sin calcular visualizaciones
                    if not live_feed_enabled:
                        # Guardar en historial si está habilitado (solo step, sin visualización completa)
                        if g_state.get('history_enabled', False):
                            try:
                                # Guardar frame mínimo para historial
                                minimal_frame = {
                                    "step": current_step,
                                    "timestamp": asyncio.get_event_loop().time()
                                }
                                g_state['simulation_history'].add_frame(minimal_frame)
                            except Exception as e:
                                logging.debug(f"Error guardando frame mínimo en historial: {e}")
                        
                        # Enviar solo log ocasional (cada 100 pasos) para indicar que la simulación está corriendo
                        if current_step % 100 == 0:
                            await broadcast({
                                "type": "simulation_log",
                                "payload": f"[Simulación] Paso {current_step} (Live feed desactivado)"
                            })
                        continue
                    
                    # Validar que el motor tenga un estado válido
                    if g_state['motor'].state.psi is None:
                        logging.warning("Motor activo pero sin estado psi. Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # --- CALCULAR VISUALIZACIONES SOLO SI LIVE_FEED ESTÁ ACTIVO ---
                    # Obtener delta_psi si está disponible para visualizaciones de flujo
                    delta_psi = g_state['motor'].last_delta_psi if hasattr(g_state['motor'], 'last_delta_psi') else None
                    viz_data = get_visualization_data(
                        g_state['motor'].state.psi, 
                        g_state.get('viz_type', 'density'),
                        delta_psi=delta_psi,
                        motor=g_state['motor']
                    )
                    
                    # Validar que viz_data tenga los campos necesarios
                    if not viz_data or not isinstance(viz_data, dict):
                        logging.warning("get_visualization_data retornó datos inválidos. Saltando frame.")
                        await asyncio.sleep(0.1)
                        continue
                    
                    frame_payload = {
                        "step": current_step,
                        "map_data": viz_data.get("map_data", []),
                        "hist_data": viz_data.get("hist_data", {}),
                        "poincare_coords": viz_data.get("poincare_coords", []),
                        "phase_attractor": viz_data.get("phase_attractor"),
                        "flow_data": viz_data.get("flow_data"),
                        "phase_hsv_data": viz_data.get("phase_hsv_data"),
                        "complex_3d_data": viz_data.get("complex_3d_data"),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    
                    # Guardar en historial si está habilitado
                    if g_state.get('history_enabled', False):
                        try:
                            g_state['simulation_history'].add_frame(frame_payload)
                        except Exception as e:
                            logging.debug(f"Error guardando frame en historial: {e}")
                    
                    # Enviar frame solo si live_feed está activo
                    await broadcast({"type": "simulation_frame", "payload": frame_payload})
                    
                    # Enviar log de simulación cada 10 pasos para no saturar
                    if current_step % 10 == 0:
                        await broadcast({
                            "type": "simulation_log",
                            "payload": f"[Simulación] Paso {current_step} completado"
                        })
                    
                    # Capturar snapshot para análisis t-SNE (cada N pasos) - OPTIMIZADO
                    # Solo capturar si está habilitado y en el intervalo correcto
                    snapshot_interval = g_state.get('snapshot_interval', 500)  # Por defecto cada 500 pasos (más espaciado)
                    snapshot_enabled = g_state.get('snapshot_enabled', False)  # Deshabilitado por defecto para no afectar rendimiento
                    
                    if snapshot_enabled and current_step % snapshot_interval == 0:
                        if 'snapshots' not in g_state:
                            g_state['snapshots'] = []
                        
                        # Optimización: usar detach() antes de clone() para evitar grafo computacional
                        # y mover a CPU de forma asíncrona si es necesario
                        try:
                            psi_tensor = g_state['motor'].state.psi
                            # Detach y clonar de forma más eficiente
                            snapshot = psi_tensor.detach().cpu().clone() if hasattr(psi_tensor, 'detach') else psi_tensor.cpu().clone()
                            
                            g_state['snapshots'].append({
                                'psi': snapshot,
                                'step': current_step,
                                'timestamp': asyncio.get_event_loop().time()
                            })
                            
                            # Limitar número de snapshots almacenados (mantener últimos 500 para reducir memoria)
                            max_snapshots = g_state.get('max_snapshots', 500)
                            if len(g_state['snapshots']) > max_snapshots:
                                # Eliminar los más antiguos de forma eficiente
                                g_state['snapshots'] = g_state['snapshots'][-max_snapshots:]
                        except Exception as e:
                            # Si falla la captura, no afectar la simulación
                            logging.debug(f"Error capturando snapshot en paso {current_step}: {e}")
                    
                except Exception as e:
                    logging.error(f"Error en el bucle de simulación: {e}", exc_info=True)
                    # Continuar el bucle en lugar de detenerlo
                    await asyncio.sleep(0.1)
                    continue
            
            # Controla la velocidad de la simulación según simulation_speed y target_fps
            simulation_speed = g_state.get('simulation_speed', 1.0)
            target_fps = g_state.get('target_fps', 10.0)
            frame_skip = g_state.get('frame_skip', 0)
            
            # Calcular delay basado en FPS objetivo y velocidad
            base_fps = target_fps * simulation_speed
            sleep_time = max(0.001, 1.0 / base_fps)  # Mínimo 1ms para no saturar
            
            # Aplicar frame skip si está configurado
            if frame_skip > 0 and g_state.get('simulation_step', 0) % (frame_skip + 1) != 0:
                # Saltar frame: solo evolución, no visualización
                if not is_paused and motor:
                    try:
                        g_state['motor'].evolve_internal_state()
                        g_state['simulation_step'] = g_state.get('simulation_step', 0) + 1
                    except:
                        pass
            
            await asyncio.sleep(sleep_time)
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
        from .utils import sns_to_dict_recursive
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

async def handle_set_viz(args):
    viz_type = args.get("viz_type", "density")
    g_state['viz_type'] = viz_type
    if (ws := g_state['websockets'].get(args.get('ws_id'))):
        await send_notification(ws, f"Visualización cambiada a: {viz_type}", "info")
    # Si hay un motor activo, enviar un frame actualizado inmediatamente
    if g_state.get('motor'):
        try:
            motor = g_state['motor']
            if motor.state.psi is None:
                logging.warning("Motor activo pero sin estado psi. No se puede actualizar visualización.")
                return
            
            delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
            viz_data = get_visualization_data(motor.state.psi, viz_type, delta_psi=delta_psi, motor=motor)
            if not viz_data or not isinstance(viz_data, dict):
                logging.warning("get_visualization_data retornó datos inválidos.")
                return
            
            frame_payload = {
                "step": g_state.get('simulation_step', 0),
                "map_data": viz_data.get("map_data", []),
                "hist_data": viz_data.get("hist_data", {}),
                "poincare_coords": viz_data.get("poincare_coords", []),
                "phase_attractor": viz_data.get("phase_attractor"),
                "flow_data": viz_data.get("flow_data"),
                "phase_hsv_data": viz_data.get("phase_hsv_data"),
                "complex_3d_data": viz_data.get("complex_3d_data")
            }
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
    if not g_state.get('motor'):
        msg = "⚠️ No hay un modelo cargado. Primero debes cargar un experimento entrenado."
        logging.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return
    
    g_state['is_paused'] = False
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

    try:
        logging.info(f"Intentando cargar el experimento '{exp_name}' para [{args['ws_id']}]...")
        if ws: await send_notification(ws, f"Cargando modelo '{exp_name}'...", "info")
        
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
        
        checkpoint_path = get_latest_checkpoint(exp_name)
        if not checkpoint_path:
            msg = f"⚠️ El experimento '{exp_name}' no tiene checkpoints entrenados. Primero debes entrenar el modelo antes de poder cargarlo para inferencia."
            logging.warning(msg)
            if ws: await send_notification(ws, msg, "warning")
            return
        
        model, state_dict = load_model(config, checkpoint_path)
        if model is None:
            msg = f"❌ Error al cargar el modelo desde el checkpoint. Verifica que el checkpoint no esté corrupto."
            logging.error(msg)
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
        
        # Inicializar motor con configuración para acceso a GAMMA_DECAY (término Lindbladian)
        # Usar config como cfg para que el motor pueda acceder a GAMMA_DECAY
        g_state['motor'] = Aetheria_Motor(model, inference_grid_size, d_state, global_cfg.DEVICE, cfg=config)
        g_state['simulation_step'] = 0
        
        # --- CORRECCIÓN: La simulación queda en pausa, el usuario debe iniciarla manualmente ---
        g_state['is_paused'] = True
        
        # Enviar frame inicial inmediatamente para mostrar el estado inicial
        try:
            motor = g_state['motor']
            if motor and motor.state and motor.state.psi is not None:
                delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                viz_data = get_visualization_data(
                    motor.state.psi, 
                    g_state.get('viz_type', 'density'),
                    delta_psi=delta_psi,
                    motor=motor
                )
                if viz_data and isinstance(viz_data, dict):
                    # Validar que los datos sean válidos antes de enviar
                    map_data = viz_data.get("map_data", [])
                    if map_data and len(map_data) > 0:
                        frame_payload = {
                            "step": 0,
                            "map_data": map_data,
                            "hist_data": viz_data.get("hist_data", {}),
                            "poincare_coords": viz_data.get("poincare_coords", []),
                            "phase_attractor": viz_data.get("phase_attractor"),
                            "flow_data": viz_data.get("flow_data"),
                            "phase_hsv_data": viz_data.get("phase_hsv_data"),
                            "complex_3d_data": viz_data.get("complex_3d_data")
                        }
                        await broadcast({"type": "simulation_frame", "payload": frame_payload})
                        logging.info(f"Frame inicial enviado exitosamente para '{exp_name}'")
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
        
        if ws: await send_notification(ws, f"✅ Modelo '{exp_name}' cargado exitosamente. Presiona 'Iniciar' para comenzar la simulación.", "success")
        await broadcast({"type": "inference_status_update", "payload": {"status": "paused"}})
        logging.info(f"Modelo '{exp_name}' cargado por [{args['ws_id']}]. Simulación en pausa, esperando inicio manual.")

    except Exception as e:
        logging.error(f"Error crítico cargando experimento '{exp_name}': {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al cargar '{exp_name}': {str(e)}", "error")

async def handle_reset(args):
    ws = g_state['websockets'].get(args['ws_id'])
    if g_state.get('motor'):
        g_state['motor'].state = QuantumState(g_state['motor'].grid_size, g_state['motor'].d_state, g_state['motor'].device)
        g_state['simulation_step'] = 0
        if ws: await send_notification(ws, "Estado de simulación reiniciado.", "info")

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
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                analyze_universe_atlas,
                psi_snapshots,
                compression_dim,
                perplexity,
                n_iter
            )
        
        # Calcular métricas
        metrics = calculate_phase_map_metrics(result['coords'])
        result['metrics'] = metrics
        
        logging.info(f"Análisis Atlas del Universo completado: {len(result['coords'])} puntos, spread={metrics['spread']:.2f}")
        
        if ws:
            await send_notification(ws, f"✅ Atlas del Universo completado ({len(result['coords'])} puntos)", "success")
            await send_to_websocket(ws, "analysis_universe_atlas", result)
        
    except Exception as e:
        logging.error(f"Error en análisis Atlas del Universo: {e}", exc_info=True)
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
        if ws:
            await send_notification(ws, "🔄 Analizando Mapa Químico...", "info")
        
        # Obtener estado actual del motor
        motor = g_state.get('motor')
        if not motor or not motor.state or motor.state.psi is None:
            msg = "⚠️ No hay simulación activa. Carga un experimento y ejecuta la simulación primero."
            logging.warning(msg)
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
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                analyze_cell_chemistry,
                psi,
                n_samples,
                perplexity,
                n_iter
            )
        
        logging.info(f"Análisis Mapa Químico completado: {len(result['coords'])} células")
        
        if ws:
            await send_notification(ws, f"✅ Mapa Químico completado ({len(result['coords'])} células)", "success")
            await send_to_websocket(ws, "analysis_cell_chemistry", result)
        
    except Exception as e:
        logging.error(f"Error en análisis Mapa Químico: {e}", exc_info=True)
        if ws:
            await send_notification(ws, f"❌ Error en análisis: {str(e)}", "error")
            await send_to_websocket(ws, "analysis_cell_chemistry", {
                "error": str(e)
            })

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
    
    status_msg = "activado" if enabled else "desactivado"
    logging.info(f"Live feed {status_msg}. La simulación continuará ejecutándose {'sin calcular visualizaciones' if not enabled else 'y enviando datos en tiempo real'}.")
    
    if ws:
        await send_notification(ws, f"Live feed {status_msg}. {'La simulación corre sin calcular visualizaciones para mejor rendimiento.' if not enabled else 'Enviando datos en tiempo real.'}", "info")
    
    # Enviar actualización a todos los clientes
    await broadcast({
        "type": "live_feed_status_update",
        "payload": {"enabled": g_state['live_feed_enabled']}
    })

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
        from .history_manager import HISTORY_DIR
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
        from .history_manager import HISTORY_DIR
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
        "delete_checkpoint": handle_delete_checkpoint
    },
    "simulation": {
        "set_viz": handle_set_viz,
        "set_speed": handle_set_simulation_speed,
        "set_fps": handle_set_fps,
        "set_frame_skip": handle_set_frame_skip,
        "set_live_feed": handle_set_live_feed,
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
        "clear_snapshots": handle_clear_snapshots
    },
    "inference": {
        "play": handle_play, 
        "pause": handle_pause, 
        "load_experiment": handle_load_experiment, 
        "reset": handle_reset
    },
    "system": {
        "refresh_experiments": handle_refresh_experiments
    }
}

# --- Configuración de la App aiohttp ---

def setup_routes(app):
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    STATIC_FILES_ROOT = PROJECT_ROOT / 'frontend' / 'dist'
    
    if not STATIC_FILES_ROOT.exists() or not (STATIC_FILES_ROOT / 'index.html').exists():
        logging.critical(f"Directorio de frontend '{STATIC_FILES_ROOT}' no encontrado o incompleto. Asegúrate de haber ejecutado 'npm run build' en la carpeta 'frontend'.")
        return

    logging.info(f"Sirviendo archivos estáticos desde: {STATIC_FILES_ROOT}")
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static('/', path=STATIC_FILES_ROOT)
    
    # Ruta "catch-all" que sirve index.html. Esencial para Single Page Applications (SPA) como React.
    app.router.add_get('/{tail:.*}', lambda req: web.FileResponse(STATIC_FILES_ROOT / 'index.html'))


async def on_startup(app):
    """Crea la tarea del bucle de simulación cuando el servidor arranca."""
    app['simulation_loop'] = asyncio.create_task(simulation_loop())

async def on_shutdown(app):
    """Cancela la tarea del bucle de simulación cuando el servidor se apaga y guarda el estado."""
    logging.info("Iniciando cierre ordenado del servidor...")
    
    # Detener proceso de entrenamiento si está activo
    training_process = g_state.get('training_process')
    if training_process and training_process.returncode is None:
        logging.info("Deteniendo proceso de entrenamiento...")
        try:
            # Enviar señal SIGTERM para permitir que el proceso guarde su checkpoint
            training_process.terminate()
            # Esperar un poco para que el proceso pueda guardar
            try:
                await asyncio.wait_for(asyncio.to_thread(training_process.wait), timeout=5.0)
                logging.info("Proceso de entrenamiento detenido correctamente.")
            except asyncio.TimeoutError:
                logging.warning("El proceso de entrenamiento no respondió en 5 segundos. Forzando cierre...")
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
        logging.info("Guardando estado de simulación antes de cerrar...")
        try:
            # Pausar la simulación
            g_state['is_paused'] = True
            # Aquí podrías guardar el estado del quantum state si es necesario
            # Por ahora solo pausamos
            logging.info("Simulación pausada y lista para guardar.")
        except Exception as e:
            logging.error(f"Error al guardar estado de simulación: {e}")
    
    # Notificar a los clientes que el servidor se está cerrando
    try:
        await broadcast({
            "type": "notification",
            "payload": {
                "status": "warning",
                "message": "El servidor se está cerrando. Guardando estado..."
            }
        })
    except Exception as e:
        logging.warning(f"Error al notificar cierre: {e}")
    
    # Cancelar el bucle de simulación
    app['simulation_loop'].cancel()
    try:
        await app['simulation_loop']
    except asyncio.CancelledError:
        pass
    
    logging.info("Cierre ordenado completado.")

async def main():
    """Función principal para configurar e iniciar el servidor web."""
    app = web.Application()
    
    setup_routes(app)
    
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, global_cfg.LAB_SERVER_HOST, global_cfg.LAB_SERVER_PORT)
    
    logging.info(f"Servidor Aetheria listo y escuchando en http://{global_cfg.LAB_SERVER_HOST}:{global_cfg.LAB_SERVER_PORT}")
    await site.start()
    
    # Mantiene el servidor corriendo indefinidamente
    await asyncio.Event().wait()
