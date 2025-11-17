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

        # --- LA CORRECCIÓN CLAVE ESTÁ AQUÍ ---
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
        from .utils import check_and_create_dir
        from . import config as global_cfg
        
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
                        # Formato: "Episodio 100/1100 | Loss: 4.599322 | Reward: -4.599322 | Quietud: ... | Complejidad: ..."
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
            from .utils import get_experiment_list
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