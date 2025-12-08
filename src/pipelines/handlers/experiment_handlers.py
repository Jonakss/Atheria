"""Handlers para gestión de experimentos."""
import asyncio
import logging
import os
import glob
from datetime import datetime
from ... import config as global_cfg
from ...server.server_state import g_state, broadcast, send_notification, send_to_websocket
from ...utils import get_experiment_list, load_experiment_config, get_latest_checkpoint
from ...server.server_handlers import create_experiment_handler

logger = logging.getLogger(__name__)


async def handle_create_experiment(args):
    """Crea un nuevo experimento de entrenamiento."""
    args['CONTINUE_TRAINING'] = False
    asyncio.create_task(create_experiment_handler(args))


async def handle_continue_experiment(args):
    """Continúa el entrenamiento de un experimento existente."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    
    if not exp_name:
        if ws: await send_notification(ws, "❌ El nombre del experimento es obligatorio.", "error")
        return
    
    try:
        config = load_experiment_config(exp_name)
        if not config:
            msg = f"❌ No se encontró la configuración para '{exp_name}'. Asegúrate de que el experimento existe."
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        continue_args = {
            'ws_id': args.get('ws_id'),
            'EXPERIMENT_NAME': exp_name,
            'MODEL_ARCHITECTURE': getattr(config, 'MODEL_ARCHITECTURE', None),
            'LR_RATE_M': getattr(config, 'LR_RATE_M', None),
            'GRID_SIZE_TRAINING': getattr(config, 'GRID_SIZE_TRAINING', None),
            'QCA_STEPS_TRAINING': getattr(config, 'QCA_STEPS_TRAINING', None),
            'CONTINUE_TRAINING': True,
        }
        
        required_fields = ['MODEL_ARCHITECTURE', 'LR_RATE_M', 'GRID_SIZE_TRAINING', 'QCA_STEPS_TRAINING']
        missing_fields = [field for field in required_fields if continue_args[field] is None]
        if missing_fields:
            msg = f"❌ La configuración del experimento '{exp_name}' está incompleta. Faltan: {', '.join(missing_fields)}"
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        episodes_to_add = args.get('EPISODES_TO_ADD')
        if episodes_to_add:
            checkpoint_path = get_latest_checkpoint(exp_name)
            if checkpoint_path:
                try:
                    import torch
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    current_episode = checkpoint.get('episode', config.TOTAL_EPISODES)
                    continue_args['TOTAL_EPISODES'] = current_episode + episodes_to_add
                    logging.info(f"Continuando desde episodio {current_episode}, añadiendo {episodes_to_add} más.")
                except Exception as e:
                    logging.warning(f"No se pudo leer el episodio del checkpoint: {e}")
                    continue_args['TOTAL_EPISODES'] = config.TOTAL_EPISODES + episodes_to_add
            else:
                continue_args['TOTAL_EPISODES'] = config.TOTAL_EPISODES + episodes_to_add
        else:
            continue_args['TOTAL_EPISODES'] = config.TOTAL_EPISODES
        
        from ...utils import sns_to_dict_recursive
        if hasattr(config, 'MODEL_PARAMS') and config.MODEL_PARAMS is not None:
            model_params = config.MODEL_PARAMS
            continue_args['MODEL_PARAMS'] = sns_to_dict_recursive(model_params)
            if not continue_args['MODEL_PARAMS'] or (isinstance(continue_args['MODEL_PARAMS'], dict) and len(continue_args['MODEL_PARAMS']) == 0):
                msg = f"❌ MODEL_PARAMS está vacío en la configuración de '{exp_name}'."
                logging.error(msg)
                if ws: await send_notification(ws, msg, "error")
                return
        else:
            msg = f"❌ No se encontró MODEL_PARAMS en la configuración de '{exp_name}'."
            logging.error(msg)
            if ws: await send_notification(ws, msg, "error")
            return
        
        asyncio.create_task(create_experiment_handler(continue_args))
        
    except Exception as e:
        logging.error(f"Error al continuar el entrenamiento de '{exp_name}': {e}", exc_info=True)
        if ws: await send_notification(ws, f"❌ Error al continuar el entrenamiento: {str(e)}", "error")


async def handle_stop_training(args):
    """Detiene el entrenamiento en curso."""
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


async def handle_delete_experiment(args):
    """Elimina un experimento completo."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    
    if not exp_name:
        if ws: await send_notification(ws, "Nombre de experimento no proporcionado.", "error")
        return
    
    try:
        import shutil
        
        exp_dir = os.path.join(global_cfg.EXPERIMENTS_DIR, exp_name)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            logging.info(f"Directorio de experimento eliminado: {exp_dir}")
        
        checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logging.info(f"Directorio de checkpoints eliminado: {checkpoint_dir}")
        
        active_exp = g_state.get('active_experiment')
        if g_state.get('motor') and active_exp == exp_name:
            g_state['motor'] = None
            g_state['active_experiment'] = None
            g_state['simulation_step'] = 0
            g_state['is_paused'] = True
            from ..core.status_helpers import build_inference_status_payload
            status_payload = build_inference_status_payload("paused")
            await broadcast({"type": "inference_status_update", "payload": status_payload})
        
        if ws: await send_notification(ws, f"✅ Experiment '{exp_name}' eliminado exitosamente.", "success")
        await handle_refresh_experiments(args)
        
    except Exception as e:
        logging.error(f"Error al eliminar experimento '{exp_name}': {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al eliminar experimento: {str(e)}", "error")


async def handle_list_checkpoints(args):
    """Lista todos los checkpoints de un experimento."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    
    if not exp_name:
        if ws: await send_notification(ws, "Nombre de experimento no proporcionado.", "error")
        return
    
    try:
        checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        if not os.path.exists(checkpoint_dir):
            if ws: await send_to_websocket(ws, "checkpoints_list", {"checkpoints": []})
            return
        
        checkpoints = []
        for checkpoint_file in glob.glob(os.path.join(checkpoint_dir, "*.pth")):
            try:
                stat = os.stat(checkpoint_file)
                filename = os.path.basename(checkpoint_file)
                
                episode = 0
                is_best = 'best' in filename.lower()
                if 'ep' in filename:
                    try:
                        episode = int(filename.split('ep')[-1].split('.')[0])
                    except:
                        pass
                
                # Check if associated snapshot exists
                snapshot_filename = f"snapshot_ep{episode}.pt"
                snapshot_path = os.path.join(checkpoint_dir, snapshot_filename)
                has_snapshot = os.path.exists(snapshot_path)

                checkpoints.append({
                    "filename": filename,
                    "episode": episode,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_best": is_best,
                    "has_snapshot": has_snapshot
                })
            except Exception as e:
                logging.warning(f"Error procesando checkpoint {checkpoint_file}: {e}")
        
        checkpoints.sort(key=lambda x: x['episode'], reverse=True)
        
        if ws: await send_to_websocket(ws, "checkpoints_list", {"checkpoints": checkpoints})
        
    except Exception as e:
        logging.error(f"Error al listar checkpoints: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al listar checkpoints: {str(e)}", "error")


async def handle_load_checkpoint_snapshot(args):
    """Carga el snapshot asociado a un checkpoint de entrenamiento."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    episode = args.get("EPISODE")
    
    if not exp_name or episode is None:
        if ws: await send_notification(ws, "Faltan parámetros requeridos (EXPERIMENT_NAME, EPISODE).", "error")
        return

    try:
        motor = g_state.get('motor')
        if not motor:
             if ws: await send_notification(ws, "⚠️ No hay un motor activo.", "warning")
             return

        checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        snapshot_filename = f"snapshot_ep{episode}.pt"
        snapshot_path = os.path.join(checkpoint_dir, snapshot_filename)
        
        if not os.path.exists(snapshot_path):
            if ws: await send_notification(ws, f"❌ No se encontró snapshot para el episodio {episode}.", "error")
            return

        # Pausar simulación
        was_paused = g_state.get('is_paused', True)
        if not was_paused:
            g_state['is_paused'] = True
            if ws: await send_notification(ws, "Pausando para cargar snapshot...", "info")
            # Enviar actualización de estado
            from ..core.status_helpers import build_inference_status_payload
            await broadcast({"type": "inference_status_update", "payload": build_inference_status_payload("paused")})

        import torch
        # Cargar el tensor
        psi = torch.load(snapshot_path, map_location=motor.device)
        
        # Validar dimensión
        # (Opcional: chequear si coincide con d_state del motor, pero confiamos por ahora)
        
        # Inyectar en motor
        if hasattr(motor, 'set_dense_state'):
            motor.set_dense_state(psi)
        elif hasattr(motor, 'state'):
             motor.state.psi = psi
        else:
             raise NotImplementedError("El motor no soporta setear estado explícitamente.")
        
        # Actualizar step
        g_state['simulation_step'] = 0 # No sabemos el step real absoluto, o asumimos que es del episodio?
        # En realidad snapshots de entrenamiento son de episodios.
        # Podríamos usar steps = episode * qca_steps_per_episode si tuviéramos esa info.
        
        msg = f"✅ Snapshot de episodio {episode} cargado."
        logging.info(msg)
        if ws: await send_notification(ws, msg, "success")
        
    except Exception as e:
        logging.error(f"Error cargando snapshot de entrenamiento: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error cargando snapshot: {str(e)}", "error")

async def handle_delete_checkpoint(args):
    """Elimina un checkpoint específico."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("EXPERIMENT_NAME")
    filename = args.get("FILENAME")
    
    if not exp_name or not filename:
        if ws: await send_notification(ws, "Faltan parámetros (EXPERIMENT_NAME, FILENAME).", "error")
        return

    try:
        checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name)
        file_path = os.path.join(checkpoint_dir, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            # También intentar borrar el snapshot asociado si existe
            # Checkpoint name usually: checkpoint_ep10.pth
            # Snapshot name usually: snapshot_ep10.pt
            if 'ep' in filename and filename.endswith('.pth'):
                try:
                    episode_str = filename.split('ep')[-1].split('.')[0]
                    snapshot_path = os.path.join(checkpoint_dir, f"snapshot_ep{episode_str}.pt")
                    if os.path.exists(snapshot_path):
                        os.remove(snapshot_path)
                        logging.info(f"Snapshot asociado eliminado: {snapshot_path}")
                except:
                    pass
            
            msg = f"✅ Checkpoint '{filename}' eliminado."
            logging.info(msg)
            if ws: await send_notification(ws, msg, "success")
            await handle_list_checkpoints(args)
        else:
            if ws: await send_notification(ws, f"No se encontró el archivo {filename}.", "error")
            
    except Exception as e:
        logging.error(f"Error eliminando checkpoint: {e}")
        if ws: await send_notification(ws, f"Error eliminando checkpoint: {str(e)}", "error")

async def handle_cleanup_checkpoints(args):
    """Elimina checkpoints antiguos, manteniendo solo los mejores."""
    # TODO: Implementar lógica de limpieza inteligente (ej: mantener ultimos 5 + todos los 'best')
    # Por ahora solo un placeholder
    ws = g_state['websockets'].get(args.get('ws_id'))
    if ws: await send_notification(ws, "Limpieza automática aún no implementada.", "warning")

async def handle_refresh_experiments(args):
    """Refresca la lista de experimentos y la envía al cliente."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    try:
        experiments = get_experiment_list()
        if ws: await send_to_websocket(ws, "experiments_list", {"experiments": experiments})
    except Exception as e:
        logging.error(f"Error refrescando experimentos: {e}")

async def handle_list_quantum_models(args):
    """Lista modelos cuánticos disponibles (placeholder)."""
    # Esta función parece que faltaba o se movió.
    # Implementación básica para evitar crash.
    ws = g_state['websockets'].get(args.get('ws_id'))
    # TODO: Implementar listado real si es necesario
    if ws: await send_to_websocket(ws, "quantum_models_list", {"models": []})


# Diccionario de handlers para esta categoría
HANDLERS = {
    "create": handle_create_experiment,
    "continue": handle_continue_experiment,
    "stop": handle_stop_training,
    "delete": handle_delete_experiment,
    "list_checkpoints": handle_list_checkpoints,
    "delete_checkpoint": handle_delete_checkpoint,
    "cleanup_checkpoints": handle_cleanup_checkpoints,
    "refresh_experiments": handle_refresh_experiments,
    "list_quantum_models": handle_list_quantum_models,
    "load_checkpoint_snapshot": handle_load_checkpoint_snapshot,
}

