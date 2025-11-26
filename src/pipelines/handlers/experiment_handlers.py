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
                
                checkpoints.append({
                    "filename": filename,
                    "episode": episode,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_best": is_best
                })
            except Exception as e:
                logging.warning(f"Error procesando checkpoint {checkpoint_file}: {e}")
        
        checkpoints.sort(key=lambda x: x['episode'], reverse=True)
        
        if ws: await send_to_websocket(ws, "checkpoints_list", {"checkpoints": checkpoints})
        
    except Exception as e:
        logging.error(f"Error al listar checkpoints: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al listar checkpoints: {str(e)}", "error")


async def handle_cleanup_checkpoints(args):
    """Limpia checkpoints antiguos de un experimento."""
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
                    ep_num = int(f.replace("checkpoint_ep", "").replace(".pth", ""))
                    checkpoints.append((ep_num, full_path))
                except ValueError:
                    continue
        
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        best_checkpoint = None
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        if os.path.exists(best_path):
            best_checkpoint = best_path
        
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
        checkpoint_path = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_name, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            if ws: await send_notification(ws, f"Checkpoint '{checkpoint_name}' no encontrado.", "error")
            return
        
        os.remove(checkpoint_path)
        if ws: await send_notification(ws, f"✅ Checkpoint '{checkpoint_name}' eliminado.", "success")
        
        await handle_list_checkpoints({'ws_id': args.get('ws_id'), 'EXPERIMENT_NAME': exp_name})
        
    except Exception as e:
        logging.error(f"Error al eliminar checkpoint: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error al eliminar checkpoint: {str(e)}", "error")


async def handle_refresh_experiments(args):
    """Refresca la lista de experimentos."""
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


# Diccionario de handlers para esta categoría
HANDLERS = {
    "create": handle_create_experiment,
    "continue": handle_continue_experiment,
    "stop": handle_stop_training,
    "delete": handle_delete_experiment,
    "list_checkpoints": handle_list_checkpoints,
    "delete_checkpoint": handle_delete_checkpoint,
    "cleanup_checkpoints": handle_cleanup_checkpoints,
    "refresh_experiments": handle_refresh_experiments,  # También usado en system
}

