# src/pipelines/handlers/snapshot_handlers.py
"""
Handlers para gestionar Snapshots de Inferencia (guardar, cargar, listar).
"""
import logging
import torch
from ...server.server_state import g_state, send_notification, send_to_websocket
from ...managers import snapshot_manager

logger = logging.getLogger(__name__)

async def handle_save_snapshot(args):
    """Guarda el estado actual de la simulación."""
    ws_id = args.get('ws_id')
    ws = g_state['websockets'].get(ws_id)
    motor = g_state.get('motor')
    experiment_name = g_state.get('active_experiment')
    step = g_state.get('simulation_step', 0)

    if not all([ws, motor, experiment_name]):
        msg = "⚠️ No se puede guardar. Se requiere una simulación activa."
        logger.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return

    try:
        # Pausar la simulación para garantizar un estado consistente
        was_paused = g_state.get('is_paused', True)
        if not was_paused:
            g_state['is_paused'] = True
            await send_notification(ws, "Pausando para guardar snapshot...", "info")

        # Obtener el estado (psi) del motor
        psi = None
        if hasattr(motor, 'native_engine') and motor.native_engine:
            # Para motor nativo, necesitamos obtener el estado denso
            psi = motor.get_dense_state()
        elif hasattr(motor, 'state') and hasattr(motor.state, 'psi'):
            # Para motor Python
            psi = motor.state.psi

        if psi is None or not isinstance(psi, torch.Tensor):
            raise ValueError("No se pudo obtener un tensor de estado válido desde el motor.")

        # Guardar el snapshot
        filepath = snapshot_manager.save_snapshot(experiment_name, step, psi)

        if filepath:
            msg = f"✅ Snapshot guardado en el paso {step}."
            logger.info(msg)
            await send_notification(ws, msg, "success")
            # Opcional: enviar la lista actualizada de snapshots al frontend
            await handle_list_snapshots({'ws_id': ws_id, 'experiment_name': experiment_name})
        else:
            raise RuntimeError("La función save_snapshot falló y no retornó una ruta.")

    except Exception as e:
        logger.error(f"Error al guardar el snapshot: {e}", exc_info=True)
        if ws: await send_notification(ws, f"❌ Error al guardar: {str(e)}", "error")

    finally:
        # Reanudar la simulación si estaba en ejecución
        if not was_paused:
            g_state['is_paused'] = False


async def handle_list_snapshots(args):
    """Lista los snapshots disponibles para un experimento."""
    ws_id = args.get('ws_id')
    ws = g_state['websockets'].get(ws_id)
    experiment_name = args.get('experiment_name') or g_state.get('active_experiment')

    if not experiment_name:
        if ws: await send_notification(ws, "⚠️ No hay un experimento seleccionado.", "warning")
        return

    try:
        snapshots = snapshot_manager.list_snapshots(experiment_name)
        if ws:
            await send_to_websocket(ws, "snapshot_list", {
                "experiment_name": experiment_name,
                "snapshots": snapshots
            })
    except Exception as e:
        logger.error(f"Error al listar snapshots para '{experiment_name}': {e}", exc_info=True)
        if ws: await send_notification(ws, f"❌ Error al listar snapshots: {str(e)}", "error")


async def handle_load_snapshot(args):
    """Carga un snapshot y restaura el estado de la simulación."""
    ws_id = args.get('ws_id')
    ws = g_state['websockets'].get(ws_id)
    snapshot_path = args.get('filepath_pt')
    motor = g_state.get('motor')

    if not all([ws, motor, snapshot_path]):
        msg = "⚠️ No se puede cargar. Se requiere una simulación activa y una ruta de snapshot."
        logger.warning(msg)
        if ws: await send_notification(ws, msg, "warning")
        return

    try:
        # Pausar la simulación antes de cargar
        g_state['is_paused'] = True
        await send_notification(ws, "Cargando snapshot...", "info")

        # Cargar los datos del snapshot
        psi, metadata = snapshot_manager.load_snapshot(snapshot_path)
        if not isinstance(psi, torch.Tensor) or not metadata:
            raise ValueError("El snapshot cargado es inválido.")

        new_step = metadata.get('step', 0)

        # Mover el tensor al dispositivo correcto del motor
        device = motor.device if hasattr(motor, 'device') else 'cpu'
        psi = psi.to(device)

        # Restaurar el estado en el motor
        if hasattr(motor, 'native_engine') and motor.native_engine:
            # Para motor nativo, usamos una función para setear el estado
            motor.set_dense_state(psi)
        elif hasattr(motor, 'state'):
            # Para motor Python
            motor.state.psi = psi
        else:
            raise NotImplementedError("El motor actual no soporta la restauración de estado.")

        # Actualizar el contador de pasos global
        g_state['simulation_step'] = new_step

        msg = f"✅ Snapshot cargado. Simulación restaurada al paso {new_step}."
        logger.info(msg)
        await send_notification(ws, msg, "success")

        # Aquí podrías enviar un frame de actualización al frontend si es necesario

    except Exception as e:
        logger.error(f"Error al cargar el snapshot: {e}", exc_info=True)
        if ws: await send_notification(ws, f"❌ Error al cargar snapshot: {str(e)}", "error")

HANDLERS = {
    "save_snapshot": handle_save_snapshot,
    "list_snapshots": handle_list_snapshots,
    "load_snapshot": handle_load_snapshot,
}
