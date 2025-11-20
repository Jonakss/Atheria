# src/server_handlers.py
import asyncio
import json
import logging
import sys
from .server_state import g_state, broadcast, send_to_websocket, send_notification

async def create_experiment_handler(args):
    """
    Construye y lanza el proceso de entrenamiento de forma robusta,
    pasando los parámetros complejos como un string JSON.
    """
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    try:
        exp_name = args.get("EXPERIMENT_NAME")
        if not exp_name:
            if ws: await send_notification(ws, "El nombre del experimento es obligatorio.", "error")
            return

        await broadcast({"type": "training_status_update", "payload": {"status": "running"}})
        
        # Mensaje de notificación mejorado
        load_from = args.get("LOAD_FROM_EXPERIMENT")
        if load_from:
            if ws: await send_notification(ws, f"🚀 Iniciando entrenamiento progresivo para '{exp_name}' usando pesos de '{load_from}'...", "info")
        else:
            if ws: await send_notification(ws, f"Iniciando entrenamiento para '{exp_name}'...", "info")

        model_arch = args.get("MODEL_ARCHITECTURE")
        if not model_arch:
            if ws: await send_notification(ws, "❌ La arquitectura del modelo es obligatoria.", "error")
            return
        
        # Validar que todos los parámetros requeridos estén presentes
        required_params = ["LR_RATE_M", "GRID_SIZE_TRAINING", "QCA_STEPS_TRAINING", "TOTAL_EPISODES", "MODEL_PARAMS"]
        missing_params = [p for p in required_params if args.get(p) is None]
        if missing_params:
            if ws: await send_notification(ws, f"❌ Faltan parámetros requeridos: {', '.join(missing_params)}", "error")
            return
        
        # Guardar la configuración del experimento ANTES de iniciar el entrenamiento
        from ..utils import check_and_create_dir
        from .. import config as global_cfg
        
        exp_config = {
            "EXPERIMENT_NAME": exp_name,
            "MODEL_ARCHITECTURE": model_arch,
            "LR_RATE_M": args.get("LR_RATE_M"),
            "GRID_SIZE_TRAINING": args.get("GRID_SIZE_TRAINING"),
            "QCA_STEPS_TRAINING": args.get("QCA_STEPS_TRAINING"),
            "TOTAL_EPISODES": args.get("TOTAL_EPISODES"),
            "MODEL_PARAMS": args.get("MODEL_PARAMS", {}),
            "LOAD_FROM_EXPERIMENT": args.get("LOAD_FROM_EXPERIMENT"),  # Para transfer learning
            "GAMMA_DECAY": args.get("GAMMA_DECAY", getattr(global_cfg, 'GAMMA_DECAY', 0.01)),  # Término Lindbladian (decaimiento)
            "INITIAL_STATE_MODE_INFERENCE": args.get("INITIAL_STATE_MODE_INFERENCE", getattr(global_cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise'))  # Modo de inicialización del estado
        }
        check_and_create_dir(exp_config)
        
        command = [
            sys.executable, "-m", "src.trainer",
            "--experiment_name", exp_name,
            "--model_architecture", model_arch,
            "--lr_rate_m", str(args.get("LR_RATE_M")),
            "--grid_size_training", str(args.get("GRID_SIZE_TRAINING")),
            "--qca_steps_training", str(args.get("QCA_STEPS_TRAINING")),
            "--total_episodes", str(args.get("TOTAL_EPISODES")),
            # Serializamos el diccionario MODEL_PARAMS a un string JSON
            "--model_params", json.dumps(args.get("MODEL_PARAMS", {}))
        ]
        
        if args.get('CONTINUE_TRAINING', False):
            command.append("--continue_training")

        logging.info(f"Ejecutando comando de entrenamiento: {' '.join(command)}")

        # Lanzamos el proceso
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        g_state['training_process'] = process

        # Leer stdout y stderr en paralelo mientras el proceso corre
        async def read_stdout():
            try:
                import re
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        if process.returncode is not None:
                            break
                        await asyncio.sleep(0.01)  # Pequeña pausa para no saturar
                        continue
                    log_msg = line.decode().strip()
                    if log_msg:
                        await broadcast({"type": "training_log", "payload": f"[Entrenamiento] {log_msg}"})
                        logging.info(f"Entrenamiento stdout: {log_msg}")
                        
                        # Parsear mensajes de progreso del entrenamiento
                        progress_match = re.search(
                            r'Episodio\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+)(?:\s+\|\s+Reward:\s+([\d.-]+))?',
                            log_msg
                        )
                        if progress_match:
                            current_episode = int(progress_match.group(1))
                            total_episodes = int(progress_match.group(2))
                            loss = float(progress_match.group(3))
                            reward = float(progress_match.group(4)) if progress_match.group(4) else None
                            
                            # Enviar progreso al frontend
                            progress_payload = {
                                "current_episode": current_episode,
                                "total_episodes": total_episodes,
                                "avg_loss": loss
                            }
                            if reward is not None:
                                progress_payload["avg_reward"] = reward
                            
                            await broadcast({
                                "type": "training_progress",
                                "payload": progress_payload
                            })
            except Exception as e:
                logging.error(f"Error leyendo stdout: {e}")
        
        async def read_stderr():
            try:
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        if process.returncode is not None:
                            break
                        await asyncio.sleep(0.01)  # Pequeña pausa para no saturar
                        continue
                    error_msg = line.decode().strip()
                    if error_msg:
                        await broadcast({"type": "training_log", "payload": f"[Error] {error_msg}"})
                        logging.error(f"Error en entrenamiento: {error_msg}")
            except Exception as e:
                logging.error(f"Error leyendo stderr: {e}")
        
        # Crear tareas para leer stdout y stderr
        stdout_task = asyncio.create_task(read_stdout())
        stderr_task = asyncio.create_task(read_stderr())
        
        # Esperar a que el proceso termine
        return_code = await process.wait()
        
        # Esperar un poco más para que las tareas de lectura terminen de leer
        await asyncio.sleep(0.5)
        
        # Cancelar las tareas de lectura si aún están corriendo
        stdout_task.cancel()
        stderr_task.cancel()
        
        try:
            await stdout_task
        except asyncio.CancelledError:
            pass
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass
        
        logging.info(f"Proceso de entrenamiento para '{exp_name}' finalizado con código {return_code}.")

        if return_code != 0:
            # Intentar leer stderr completo si hay error
            try:
                remaining_stderr = await process.stderr.read()
                if remaining_stderr:
                    error_message = remaining_stderr.decode().strip()
                    await broadcast({"type": "training_log", "payload": f"[Error] {error_message}"})
                    logging.error(f"Error completo en el script de entrenamiento: {error_message}")
            except Exception as e:
                logging.error(f"Error al leer stderr restante: {e}")
            
            if ws: await send_notification(ws, f"❌ El entrenamiento falló (código {return_code}). Revisa los logs para más detalles.", "error")
        else:
            if ws: await send_notification(ws, f"✅ Entrenamiento para '{exp_name}' completado exitosamente.", "success")
            
            # Actualizar la lista de experimentos para reflejar los nuevos checkpoints
            from ..utils import get_experiment_list
            try:
                updated_experiments = get_experiment_list()
                await broadcast({
                    "type": "initial_state",
                    "payload": {
                        "experiments": updated_experiments,
                        "training_status": "idle",
                        "inference_status": "paused" if g_state.get('is_paused', True) else "running"
                    }
                })
                logging.info(f"Lista de experimentos actualizada después del entrenamiento de '{exp_name}'")
            except Exception as e:
                logging.warning(f"Error al actualizar lista de experimentos: {e}")

    except Exception as e:
        logging.error(f"Error al lanzar el handler de entrenamiento: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error crítico del servidor al iniciar el entrenamiento.", "error")
    finally:
        g_state['training_process'] = None
        await broadcast({"type": "training_status_update", "payload": {"status": "idle"}})

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
        from ..utils import load_experiment_config, get_latest_checkpoint
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
            if motor.state.psi is None:
                logging.warning("Motor activo pero sin estado psi. No se puede actualizar visualización.")
                return
            
            from .pipeline_viz import get_visualization_data
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
    """Carga un experimento para inferencia, manejando casos especiales como MLP."""
    ws = g_state['websockets'].get(args['ws_id'])
    exp_name = args.get("experiment_name")
    if not exp_name:
        if ws: await send_notification(ws, "Nombre de experimento no proporcionado.", "error")
        return

    try:
        logging.info(f"Intentando cargar el experimento '{exp_name}' para [{args['ws_id']}]...")
        if ws: await send_notification(ws, f"Cargando modelo '{exp_name}'...", "info")
        
        from ..utils import load_experiment_config, get_latest_checkpoint
        from . import config as global_cfg
        from .model_loader import load_model
        from ..engines.qca_engine import Aetheria_Motor
        from ..managers.roi_manager import ROIManager
        
        config = load_experiment_config(exp_name)
        if not config:
            msg = f"❌ No se encontró la configuración para '{exp_name}'."
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        # Asegurar valores por defecto
        if not hasattr(config, 'GAMMA_DECAY') or getattr(config, 'GAMMA_DECAY', None) is None:
            config.GAMMA_DECAY = getattr(global_cfg, 'GAMMA_DECAY', 0.01)
        
        if not hasattr(config, 'INITIAL_STATE_MODE_INFERENCE') or getattr(config, 'INITIAL_STATE_MODE_INFERENCE', None) is None:
            config.INITIAL_STATE_MODE_INFERENCE = getattr(global_cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
        
        checkpoint_path = get_latest_checkpoint(exp_name)
        if not checkpoint_path:
            msg = f"⚠️ '{exp_name}' no tiene checkpoints. Entrena primero."
            logging.warning(msg)
            if ws: await send_notification(ws, msg, "warning")
            return
        
        model, state_dict = load_model(config, checkpoint_path)
        if model is None:
            msg = f"❌ Error al cargar el modelo."
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        model.eval()
        
        d_state = config.MODEL_PARAMS.d_state
        
        # --- VALIDACIÓN ESPECIAL PARA MLP ---
        # Los MLPs no pueden cambiar de tamaño de grid entre entrenamiento e inferencia
        training_grid_size = getattr(config, 'GRID_SIZE_TRAINING', global_cfg.GRID_SIZE_TRAINING)
        inference_grid_size = global_cfg.GRID_SIZE_INFERENCE
        model_arch = getattr(config, 'MODEL_ARCHITECTURE', 'UNET')
        
        if model_arch == 'MLP' and training_grid_size != inference_grid_size:
            msg = f"⚠️ Modelos MLP requieren grid fijo. Cambiando inferencia de {inference_grid_size} a {training_grid_size}."
            logging.warning(msg)
            if ws: await send_notification(ws, msg, "warning")
            inference_grid_size = training_grid_size
        elif training_grid_size and training_grid_size != inference_grid_size:
             logging.info(f"Escalando grid de entrenamiento ({training_grid_size}) a inferencia ({inference_grid_size})")
        
        # Inicializar motor
        g_state['motor'] = Aetheria_Motor(model, inference_grid_size, d_state, global_cfg.DEVICE, cfg=config)
        g_state['simulation_step'] = 0
        
        # Compilación opcional
        try:
            g_state['motor'].compile_model()
            if g_state['motor'].is_compiled:
                if ws: await send_notification(ws, "✅ Modelo compilado con torch.compile()", "info")
        except Exception as e:
            logging.warning(f"No se pudo compilar: {e}")
        
        # Actualizar ROI manager con el tamaño correcto
        roi_manager = g_state.get('roi_manager')
        if roi_manager:
            roi_manager.grid_size = inference_grid_size
            roi_manager.clear_roi()
        else:
            g_state['roi_manager'] = ROIManager(grid_size=inference_grid_size)
        
        # Simulación en pausa al cargar
        g_state['is_paused'] = True
        
        # Enviar frame inicial si live feed está activo
        if g_state.get('live_feed_enabled', True):
            try:
                from .pipeline_viz import get_visualization_data
                motor = g_state['motor']
                delta_psi = getattr(motor, 'last_delta_psi', None)
                viz_data = get_visualization_data(
                    motor.state.psi, 
                    g_state.get('viz_type', 'density'),
                    delta_psi=delta_psi,
                    motor=motor
                )
                
                if viz_data and isinstance(viz_data, dict) and viz_data.get("map_data"):
                    frame_payload = {
                        "step": 0,
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
                logging.error(f"Error frame inicial: {e}")
        
        compile_status = {
            "is_compiled": g_state['motor'].is_compiled,
            "model_name": model.__class__.__name__,
            "compiles_enabled": getattr(model, '_compiles', True)
        }
        
        if ws: await send_notification(ws, f"✅ Modelo '{exp_name}' cargado.", "success")
        await broadcast({
            "type": "inference_status_update", 
            "payload": {
                "status": "paused",
                "model_loaded": True,
                "experiment_name": exp_name,
                "compile_status": compile_status
            }
        })

    except Exception as e:
        logging.error(f"Error crítico cargando experimento '{exp_name}': {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al cargar '{exp_name}': {str(e)}", "error")

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
        from . import config as global_cfg
        from ..engines.qca_engine import QuantumState
        
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
                from .pipeline_viz import get_visualization_data
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
                        logging.info("Frame de reinicio enviado exitosamente")
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

async def handle_refresh_experiments(args):
    """Refresca la lista de experimentos y la envía a todos los clientes conectados."""
    try:
        from ..utils import get_experiment_list
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
        from . import config as global_cfg
        
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
        from . import config as global_cfg
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
        from . import config as global_cfg
        
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
        from .analysis import analyze_universe_atlas, calculate_phase_map_metrics
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
        from .analysis import analyze_cell_chemistry
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

async def handle_set_inference_config(args):
    """Configura parámetros de inferencia (requiere recargar experimento para algunos)."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    # Parámetros que requieren recargar el experimento
    grid_size = args.get("grid_size")
    initial_state_mode = args.get("initial_state_mode")
    gamma_decay = args.get("gamma_decay")
    
    changes = []
    
    if grid_size is not None:
        from . import config as global_cfg
        old_size = global_cfg.GRID_SIZE_INFERENCE
        global_cfg.GRID_SIZE_INFERENCE = int(grid_size)
        changes.append(f"Grid size: {old_size} → {grid_size}")
        logging.info(f"Grid size de inferencia configurado a: {grid_size} (requiere recargar experimento)")
    
    if initial_state_mode is not None:
        from . import config as global_cfg
        old_mode = getattr(global_cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
        global_cfg.INITIAL_STATE_MODE_INFERENCE = str(initial_state_mode)
        changes.append(f"Inicialización: {old_mode} → {initial_state_mode}")
        logging.info(f"Modo de inicialización configurado a: {initial_state_mode} (requiere recargar experimento)")
    
    if gamma_decay is not None:
        from . import config as global_cfg
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