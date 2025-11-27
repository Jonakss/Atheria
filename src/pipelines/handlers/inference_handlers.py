"""Handlers para comandos de inferencia (play, pause, load, reset, etc.)."""
import asyncio
import logging
import torch
import numpy as np
import time
import importlib
import sys
import gc
from pathlib import Path
from types import SimpleNamespace

from ...server.server_state import g_state, broadcast, send_notification, send_to_websocket, optimize_frame_payload
from ..core.status_helpers import build_inference_status_payload
from ...model_loader import load_model
from src import config as global_cfg

# ... (existing imports)

async def handle_set_inference_config(args):
    """Configura parámetros de inferencia."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    
    grid_size = args.get("grid_size")
    initial_state_mode = args.get("initial_state_mode")
    gamma_decay = args.get("gamma_decay")
    
    changes = []
    
    if grid_size is not None:
        new_size = int(grid_size)
        global_cfg.GRID_SIZE_INFERENCE = new_size
        g_state['inference_grid_size'] = new_size
        changes.append(f"Grid size: {new_size}")
        
        roi_manager = g_state.get('roi_manager')
        if roi_manager:
            roi_manager.grid_size = new_size
            roi_manager.clear_roi()
        else:
            from ...managers.roi_manager import ROIManager
            g_state['roi_manager'] = ROIManager(grid_size=new_size)
    
    if initial_state_mode is not None:
        global_cfg.INITIAL_STATE_MODE_INFERENCE = str(initial_state_mode)
        changes.append(f"Inicialización: {initial_state_mode}")
    
    if gamma_decay is not None:
        new_gamma = float(gamma_decay)
        if new_gamma < 0: new_gamma = 0.0
        elif new_gamma > 10.0: new_gamma = 10.0
        global_cfg.GAMMA_DECAY = new_gamma
        
        motor = g_state.get('motor')
        if motor:
            if hasattr(motor, 'cfg') and motor.cfg:
                motor.cfg.GAMMA_DECAY = new_gamma
            elif not hasattr(motor, 'cfg') or motor.cfg is None:
                motor.cfg = SimpleNamespace(GAMMA_DECAY=new_gamma)
        
        changes.append(f"Gamma Decay: {new_gamma}")
    
    if changes:
        msg = f"✅ Configuración actualizada: {', '.join(changes)}"
        if ws: await send_notification(ws, msg, "success")
        
        await broadcast({
            "type": "inference_config_update",
            "payload": {
                "grid_size": global_cfg.GRID_SIZE_INFERENCE,
                "initial_state_mode": global_cfg.INITIAL_STATE_MODE_INFERENCE,
                "gamma_decay": global_cfg.GAMMA_DECAY
            }
        })
        
        # Si cambió el grid_size y hay un experimento activo, recargarlo para aplicar cambios
        if grid_size is not None and g_state.get('active_experiment'):
            logging.info(f"🔄 Recargando experimento '{g_state['active_experiment']}' para aplicar nuevo grid size: {new_size}")
            if ws: await send_notification(ws, "Recargando simulación para aplicar cambio de grid...", "info")
            
            # Pequeña pausa para asegurar que el mensaje llegue
            await asyncio.sleep(0.1)
            
            await handle_load_experiment({
                'ws_id': args.get('ws_id'),
                'experiment_name': g_state['active_experiment'],
                'force_engine': g_state.get('motor_type', 'auto')
            })
from ...engines.qca_engine import Aetheria_Motor, QuantumState
from ...engines.harmonic_engine import SparseHarmonicEngine
from ..viz import get_visualization_data
from ...utils import get_latest_checkpoint, get_latest_jit_model, load_experiment_config

logger = logging.getLogger(__name__)


async def handle_play(args):
    """Inicia la simulación."""
    logging.info("🎮 handle_play() llamado - Iniciando simulación...")
    ws = g_state['websockets'].get(args.get('ws_id'))
    
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
        # Motor nativo: verificar que tenga modelo cargado usando verificaciones livianas
        # OPTIMIZACIÓN: No llamar a get_dense_state() para validación, usar model_loaded y get_matter_count
        try:
            # Verificar que el motor nativo esté inicializado
            if not hasattr(motor, 'model_loaded') or not motor.model_loaded:
                msg = "⚠️ El motor nativo no tiene un modelo cargado. Intenta recargar el experimento."
                logging.warning(msg)
                if ws: await send_notification(ws, msg, "warning")
                return
            
            # Verificación liviana del estado (O(1))
            logging.info("🔍 Verificando estado del motor nativo (verificación liviana)...")
            try:
                # Verificar si el motor nativo tiene partículas almacenadas
                matter_count = 0
                if hasattr(motor.native_engine, 'get_matter_count'):
                    matter_count = motor.native_engine.get_matter_count()
                    logging.info(f"✅ Motor nativo tiene {matter_count} partículas almacenadas")

                    # Si no hay partículas, intentar regenerar estado inicial
                    if matter_count == 0:
                        logging.warning("⚠️ Motor nativo no tiene partículas. Intentando regenerar estado inicial...")
                        if hasattr(motor, 'regenerate_initial_state'):
                            logging.info("🛠️ Regenerando estado inicial según INITIAL_STATE_MODE_INFERENCE...")
                            try:
                                loop = asyncio.get_event_loop()
                                await asyncio.wait_for(
                                    loop.run_in_executor(
                                        None,
                                        lambda: motor.regenerate_initial_state()
                                    ),
                                    timeout=15.0  # Timeout más largo para regeneración
                                )
                                logging.info(f"✅ Estado inicial regenerado")
                            except asyncio.TimeoutError:
                                logging.error("❌ Timeout regenerando estado inicial (15s).")
                                msg = "⚠️ Timeout regenerando estado. Intenta reiniciar o usa motor Python."
                                if ws: await send_notification(ws, msg, "error")
                                return
                            except Exception as e:
                                logging.error(f"❌ Error regenerando estado inicial: {e}", exc_info=True)
                                msg = "⚠️ Error regenerando estado. Intenta reiniciar o usa motor Python."
                                if ws: await send_notification(ws, msg, "error")
                                return
                else:
                    # Fallback: asumir que hay partículas si model_loaded=True
                    logging.info("✅ Motor nativo inicializado (get_matter_count no disponible)")
            
            except Exception as check_error:
                # Si la verificación liviana falla, loguear pero no detener
                logging.warning(f"⚠️ Error en verificación liviana del motor nativo: {check_error}")
                logging.info("💡 Continuando con la simulación (el motor puede estar en estado válido)")
            
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
    
    # Enviar frame inicial si es posible (mejor esfuerzo - no bloquear la simulación)
    if motor_is_native and hasattr(motor, 'get_dense_state'):
        # OPTIMIZACIÓN: Si el grid es muy grande (>128), saltar frame inicial para evitar freeze
        # El motor nativo puede tardar en la primera conversión. Mejor esperar al primer step.
        grid_size = getattr(motor, 'grid_size', 256)
        if grid_size > 128:
             logging.info(f"⏩ Saltando frame inicial para motor nativo (grid={grid_size} > 128) para evitar bloqueo.")
        else:
            try:
                # OPTIMIZACIÓN: Timeout más largo (30s) y fallback si falla
                # La visualización puede actualizarse después - no bloquear Play
                logging.info("📤 Intentando enviar frame inicial (mejor esfuerzo, no bloqueante)...")
                loop = asyncio.get_event_loop()
                psi = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: motor.get_dense_state(check_pause_callback=lambda: g_state.get('is_paused', False))
                    ),
                    timeout=30.0  # Timeout más largo para grids grandes
                )
                if psi is not None and isinstance(psi, torch.Tensor) and psi.numel() > 0:
                    psi_abs_max = psi.abs().max().item()
                    if psi_abs_max > 1e-10:
                        viz_type = g_state.get('viz_type', 'density')
                        delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                        viz_data = get_visualization_data(psi, viz_type, delta_psi=delta_psi, motor=motor)
                        
                        if viz_data and isinstance(viz_data, dict):
                            map_data = viz_data.get("map_data", [])
                            # Verificar si hay datos válidos (soporte para listas y numpy arrays)
                            has_data = False
                            if map_data is not None:
                                if isinstance(map_data, list):
                                    has_data = len(map_data) > 0
                                elif hasattr(map_data, 'size'): # Numpy array
                                    has_data = map_data.size > 0
                            
                            if has_data:
                                map_data_np = np.array(map_data) if not isinstance(map_data, np.ndarray) else map_data
                                min_val = np.min(map_data_np)
                                max_val = np.max(map_data_np)
                                range_val = max_val - min_val
                                
                                if range_val > 1e-6:
                                    current_step = g_state.get('simulation_step', 0)
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
                                            "live_feed_enabled": g_state.get('live_feed_enabled', True),
                                            "fps": g_state.get('current_fps', 0.0)
                                        }
                                    }
                                    # CRÍTICO: Optimizar payload antes de enviar (maneja numpy arrays)
                                    frame_payload = await optimize_frame_payload(
                                        frame_payload_raw,
                                        enable_compression=g_state.get('data_compression_enabled', True),
                                        downsample_factor=g_state.get('downsample_factor', 1),
                                        viz_type=viz_type
                                    )
                                    await broadcast({"type": "simulation_frame", "payload": frame_payload})
                                    logging.info(f"📤 Frame inicial enviado al frontend (step={current_step})")
            except asyncio.TimeoutError:
                # Timeout es aceptable - la visualización se actualizará en el siguiente step
                logging.warning("⏱️ Timeout enviando frame inicial (30s). La visualización se actualizará después.")
            except Exception as e:
                # Error es aceptable - la visualización se actualizará en el siguiente step
                logging.warning(f"⚠️ Error enviando frame inicial: {e}. La visualización se actualizará después.")
    
    logging.info(f"Simulación iniciada. Motor: {type(motor).__name__}, Step: {g_state.get('simulation_step', 0)}")
    
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
    
    status_payload = build_inference_status_payload("paused")
    status_payload.update({
        "step": g_state.get('simulation_step', 0),
        "simulation_info": {
            "step": g_state.get('simulation_step', 0),
            "is_paused": True,
            "live_feed_enabled": g_state.get('live_feed_enabled', True),
            "fps": g_state.get('current_fps', 0.0),
            "epoch": g_state.get('current_epoch', 0),
            "epoch_metrics": g_state.get('epoch_metrics', {})
        }
    })
    
    await broadcast({"type": "inference_status_update", "payload": status_payload})
    if ws:
        await send_notification(ws, "Simulación pausada.", "info")


async def handle_unload_model(args):
    """Descarga el modelo cargado y limpia el estado."""
    logging.info("🗑️ handle_unload_model() llamado - Descargando modelo...")
    ws = g_state['websockets'].get(args.get('ws_id'))
    motor = g_state.get('motor')
    
    if not motor:
        if ws: 
            try:
                await send_notification(ws, "⚠️ No hay modelo cargado para descargar.", "warning")
            except Exception:
                pass
        return
    
    try:
        g_state['is_paused'] = True
        await asyncio.sleep(0.1)
        experiment_name = g_state.get('active_experiment', 'Unknown')
        
        # Limpiar motor nativo
        if hasattr(motor, 'native_engine') and motor.native_engine is not None:
            try:
                if hasattr(motor, 'cleanup'):
                    motor.cleanup()
                    await asyncio.sleep(0.1)
                elif hasattr(motor.native_engine, 'clear'):
                    motor.native_engine.clear()
                    await asyncio.sleep(0.1)
            except Exception as cleanup_error:
                logging.warning(f"Error durante cleanup de motor nativo: {cleanup_error}")
        
        # Limpiar estado del motor
        if hasattr(motor, 'state') and motor.state is not None:
            try:
                motor.state.psi = None
                motor.state = None
            except Exception:
                pass
        
        motor = None
        g_state['motor'] = None
        g_state['simulation_step'] = 0
        g_state['motor_type'] = None
        g_state['motor_is_native'] = False
        g_state['active_experiment'] = None
        g_state['is_paused'] = True
        
        if 'snapshots' in g_state: g_state['snapshots'].clear()
        if 'simulation_history' in g_state: g_state['simulation_history'].clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        logging.info(f"✅ Modelo '{experiment_name}' descargado y memoria limpiada")
        
        if 'compile_status' in g_state:
            g_state['compile_status'] = None
        
        if ws and not ws.closed:
            try:
                await send_notification(ws, f"✅ Modelo descargado. Memoria limpiada.", "success")
            except Exception:
                pass
        
        status_payload = build_inference_status_payload("idle")
        await broadcast({"type": "inference_status_update", "payload": status_payload})
        
    except Exception as e:
        logging.error(f"Error descargando modelo: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error descargando modelo: {str(e)}", "error")


async def handle_load_experiment(args):
    """Carga un experimento."""
    logging.info("📦 handle_load_experiment() llamado - Cargando experimento...")
    ws = g_state['websockets'].get(args.get('ws_id'))
    exp_name = args.get("experiment_name")
    if not exp_name:
        if ws: await send_notification(ws, "Nombre de experimento no proporcionado.", "error")
        return

    device = global_cfg.DEVICE
    device_str = str(device).split(':')[0]

    try:
        logging.info(f"Intentando cargar el experimento '{exp_name}'...")
        if ws: await send_notification(ws, f"Cargando modelo '{exp_name}'...", "info")
        
        # CRÍTICO: Actualizar grid_size de inferencia si se proporciona en args
        # Esto asegura que el grid size seleccionado por el usuario se respete al cargar
        grid_size_from_args = args.get('grid_size')
        if grid_size_from_args is not None:
            new_grid_size = int(grid_size_from_args)
            logging.info(f"🔧 Actualizando grid size de inferencia: {new_grid_size}")
            global_cfg.GRID_SIZE_INFERENCE = new_grid_size
            g_state['inference_grid_size'] = new_grid_size
            
            # Actualizar ROI manager con nuevo grid size
            roi_manager = g_state.get('roi_manager')
            if roi_manager:
                roi_manager.grid_size = new_grid_size
                roi_manager.clear_roi()
            else:
                from ...managers.roi_manager import ROIManager
                g_state['roi_manager'] = ROIManager(grid_size=new_grid_size)
        
        # Pausar y limpiar motor anterior
        g_state['is_paused'] = True
        status_payload = build_inference_status_payload("paused")
        await broadcast({"type": "inference_status_update", "payload": status_payload})
        await asyncio.sleep(0.2)
        
        old_motor = g_state.get('motor')
        if old_motor is not None:
            # Reutilizar lógica de limpieza de handle_unload_model (simplificada aquí)
            if hasattr(old_motor, 'native_engine') and old_motor.native_engine is not None:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                if hasattr(old_motor, 'cleanup'):
                    old_motor.cleanup()
            
            old_motor = None
            g_state['motor'] = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        
        # Determinar motor a usar
        force_engine = args.get('force_engine')
        use_native = False
        
        if force_engine == 'native':
            use_native = True
        elif force_engine == 'python':
            use_native = False
        elif force_engine == 'harmonic':
            use_native = False
            # Harmonic engine is a special python engine
        else:
            # Auto-detectar: Respetar configuración global si no se fuerza
            if global_cfg.USE_NATIVE_ENGINE:
                try:
                    import atheria_core
                    use_native = True
                except ImportError:
                    logging.warning("USE_NATIVE_ENGINE=True pero atheria_core no está disponible. Usando Python.")
                    use_native = False
            else:
                use_native = False
        
        # Cargar modelo
        try:
            logging.info(f"Cargando modelo con motor: {'NATIVO (C++)' if use_native else 'PYTHON'}")
            
            if use_native:
                # Lógica de carga para motor nativo
                checkpoint_path = get_latest_checkpoint(exp_name)
                if not checkpoint_path:
                    raise FileNotFoundError(f"No se encontró checkpoint para {exp_name}")
                
                jit_path = get_latest_jit_model(exp_name, silent=True)
                if not jit_path:
                    # Exportar JIT si no existe
                    logging.info("Modelo JIT no encontrado, exportando...")
                    if ws: await send_notification(ws, "Exportando modelo a JIT para motor nativo...", "info")
                    # Cargar temporalmente en Python para exportar
                    from ...utils import load_experiment_config
                    exp_cfg = load_experiment_config(exp_name)
                    if not exp_cfg:
                        raise ValueError(f"No se pudo cargar configuración de {exp_name}")
                    
                    temp_model = load_model(exp_cfg, checkpoint_path)
                    if temp_model is None:
                        raise ValueError(f"No se pudo cargar modelo de {exp_name}")
                    
                    from ...engines.native_engine_wrapper import export_model_to_jit
                    d_state = exp_cfg.MODEL_PARAMS.d_state
                    grid_size = g_state.get('inference_grid_size', global_cfg.GRID_SIZE_INFERENCE)
                    jit_path = export_model_to_jit(temp_model, exp_name, (1, d_state, grid_size, grid_size))
                    del temp_model
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                from ...engines.native_engine_wrapper import NativeEngineWrapper
                from ...utils import load_experiment_config
                exp_cfg = load_experiment_config(exp_name)
                if not exp_cfg:
                    raise ValueError(f"No se pudo cargar configuración de {exp_name}")
                
                if ws: await send_notification(ws, "Inicializando motor nativo (puede tomar unos segundos)...", "info")
                
                # CRÍTICO: Ejecutar inicialización en thread pool para no bloquear event loop
                def create_native_motor():
                    motor = NativeEngineWrapper(
                        grid_size=g_state.get('inference_grid_size', global_cfg.GRID_SIZE_INFERENCE),
                        d_state=exp_cfg.MODEL_PARAMS.d_state,
                        device=device_str,
                        cfg=exp_cfg
                    )
                    success = motor.load_model(str(jit_path))
                    if not success:
                        raise ValueError(f"No se pudo cargar modelo JIT de {exp_name}")
                    return motor
                
                # Ejecutar en thread pool
                loop = asyncio.get_event_loop()
                motor = await loop.run_in_executor(None, create_native_motor)
                
                g_state['motor_is_native'] = True
                g_state['motor_type'] = 'native'
            else:
                # Motor Python (Standard or Harmonic)
                from ...utils import load_experiment_config
                exp_cfg = load_experiment_config(exp_name)
                if not exp_cfg:
                    raise ValueError(f"No se pudo cargar configuración de {exp_name}")
                
                if ws: await send_notification(ws, f"Inicializando motor {'Harmónico' if force_engine == 'harmonic' else 'Python'}...", "info")
                
                checkpoint_path = get_latest_checkpoint(exp_name)
                
                # CRÍTICO: Ejecutar inicialización en thread pool para no bloquear event loop
                def create_python_motor():
                    model = load_model(exp_cfg, checkpoint_path)
                    if model is None:
                        raise ValueError(f"No se pudo cargar modelo de {exp_name}")
                    
                    if force_engine == 'harmonic':
                        # Usar SparseHarmonicEngine
                        motor = SparseHarmonicEngine(
                            model=model,
                            d_state=exp_cfg.MODEL_PARAMS.d_state,
                            device=device,
                            grid_size=g_state.get('inference_grid_size', global_cfg.GRID_SIZE_INFERENCE)
                        )
                        # Inyectar materia inicial si es necesario
                        # Por ahora iniciamos vacío o con una semilla central
                        initial_state = torch.randn(exp_cfg.MODEL_PARAMS.d_state, device=device)
                        motor.add_matter(0, 0, 0, initial_state)
                    else:
                        # Usar Aetheria_Motor estándar
                        motor = Aetheria_Motor(
                            model_operator=model,
                            grid_size=g_state.get('inference_grid_size', global_cfg.GRID_SIZE_INFERENCE),
                            d_state=exp_cfg.MODEL_PARAMS.d_state,
                            device=device,
                            cfg=exp_cfg
                        )
                        # Aetheria_Motor no hereda de nn.Module, llamar eval en el modelo interno
                        if hasattr(motor, 'model_operator'):
                            motor.model_operator.eval()
                            
                        # Configurar estado inicial
                        initial_mode = getattr(global_cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
                        motor.state = QuantumState(
                            motor.grid_size,
                            motor.d_state,
                            motor.device,
                            initial_mode=initial_mode
                        )
                    return motor
                
                # Ejecutar en thread pool
                loop = asyncio.get_event_loop()
                motor = await loop.run_in_executor(None, create_python_motor)
                
                g_state['motor_is_native'] = False
                g_state['motor_type'] = 'harmonic' if force_engine == 'harmonic' else 'python'
            
            # Extraer paso inicial del nombre del archivo
            initial_step = 0
            try:
                # Intentar extraer de snapshot (snapshot_..._step_123.pt)
                import re
                filename = os.path.basename(str(checkpoint_path)) if checkpoint_path else ""
                step_match = re.search(r'step_(\d+)', filename)
                if step_match:
                    initial_step = int(step_match.group(1))
                else:
                    # Intentar extraer de checkpoint (checkpoint_ep123.pth)
                    ep_match = re.search(r'_ep(\d+)', filename)
                    if ep_match:
                        episode = int(ep_match.group(1))
                        # Estimar pasos basado en QCA_STEPS_TRAINING si está disponible
                        qca_steps = getattr(exp_cfg, 'QCA_STEPS_TRAINING', 100) # Default 100
                        initial_step = episode * qca_steps
            except Exception as e:
                logging.warning(f"No se pudo extraer paso inicial del archivo: {e}")
                initial_step = 0

            g_state['motor'] = motor
            g_state['active_experiment'] = exp_name
            g_state['simulation_step'] = initial_step
            g_state['initial_step'] = initial_step  # Guardar paso inicial para calcular session_steps
            g_state['start_step'] = initial_step    # Compatibilidad
            g_state['current_epoch'] = 0
            g_state['epoch_metrics'] = {}
            
            # Mostrar información de grid scaling si aplica
            try:
                from ...utils import load_experiment_config
                exp_cfg_loaded = load_experiment_config(exp_name)
                if exp_cfg_loaded:
                    training_grid_size = getattr(exp_cfg_loaded, 'GRID_SIZE_TRAINING', None)
                    inference_grid_size = g_state.get('inference_grid_size', global_cfg.GRID_SIZE_INFERENCE)
                    if training_grid_size and training_grid_size < inference_grid_size:
                        scaling_msg = f"📐 Grid escalado: {training_grid_size}x{training_grid_size} (original) → {inference_grid_size}x{inference_grid_size} (inferencia)"
                        logging.info(scaling_msg)
                        if ws: await send_notification(ws, scaling_msg, "info")
            except Exception as e:
                logging.debug(f"No se pudo mostrar info de grid scaling: {e}")

            # Notificar éxito
            msg = f"✅ Experimento '{exp_name}' cargado exitosamente ({'Nativo' if use_native else 'Python'})."
            logging.info(msg)
            if ws: await send_notification(ws, msg, "success")
            
            status_payload = build_inference_status_payload("ready")
            await broadcast({"type": "inference_status_update", "payload": status_payload})
            
        except Exception as e:
            logging.error(f"Error cargando modelo: {e}", exc_info=True)
            if ws: await send_notification(ws, f"Error cargando modelo: {str(e)}", "error")
            # Intentar fallback a Python si falló nativo
            if use_native:
                logging.info("Intentando fallback a motor Python...")
                if ws: await send_notification(ws, "Intentando fallback a motor Python...", "warning")
                args['force_engine'] = 'python'
                await handle_load_experiment(args)
            
    except Exception as e:
        logging.error(f"Error fatal en handle_load_experiment: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error fatal cargando experimento: {str(e)}", "error")


async def handle_switch_engine(args):
    """Cambia entre motor nativo (C++) y motor Python."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    target_engine = args.get('engine', 'auto')
    
    motor = g_state.get('motor')
    current_is_native = hasattr(motor, 'native_engine') if motor else False
    current_engine_type = 'native' if current_is_native else 'python'
    
    if target_engine == 'auto':
        target_engine = 'python' if current_is_native else 'native'
    elif target_engine == current_engine_type:
        if ws: await send_notification(ws, f"⚠️ Ya estás usando el motor {current_engine_type}.", "info")
        return
    
    exp_name = g_state.get('active_experiment')
    if not exp_name:
        if ws: await send_notification(ws, f"✅ Motor {target_engine} seleccionado para próxima carga.", "info")
        return
    
    # Recargar experimento con el nuevo motor
    await handle_load_experiment({
        'ws_id': args.get('ws_id'),
        'experiment_name': exp_name,
        'force_engine': target_engine
    })


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
        initial_mode = getattr(global_cfg, 'INITIAL_STATE_MODE_INFERENCE', 'complex_noise')
        
        if hasattr(motor, 'native_engine'):
            # Reiniciar motor nativo
            # TODO: Implementar reset específico para nativo si es necesario
            # Por ahora recargamos partículas o limpiamos estado
            if hasattr(motor, 'reset'):
                motor.reset()
            else:
                # Fallback: recargar experimento
                exp_name = g_state.get('active_experiment')
                await handle_load_experiment({
                    'ws_id': args.get('ws_id'),
                    'experiment_name': exp_name,
                    'force_engine': 'native'
                })
                return
        else:
            # Reiniciar motor Python
            motor.state = QuantumState(
                motor.grid_size, 
                motor.d_state, 
                motor.device,
                initial_mode=initial_mode
            )
        
        g_state['simulation_step'] = 0
        
        # Enviar frame actualizado
        live_feed_enabled = g_state.get('live_feed_enabled', True)
        if live_feed_enabled and hasattr(motor, 'state') and motor.state and motor.state.psi is not None:
            try:
                delta_psi = motor.last_delta_psi if hasattr(motor, 'last_delta_psi') else None
                viz_type = g_state.get('viz_type', 'density')
                viz_data = get_visualization_data(motor.state.psi, viz_type, delta_psi=delta_psi, motor=motor)
                
                if viz_data and isinstance(viz_data, dict):
                    map_data = viz_data.get("map_data", [])
                    if map_data:
                        frame_payload = {
                            "step": 0,
                            "timestamp": asyncio.get_event_loop().time(),
                            "map_data": map_data,
                            "simulation_info": {
                                "step": 0,
                                "is_paused": True,
                                "live_feed_enabled": live_feed_enabled,
                                "fps": g_state.get('current_fps', 0.0)
                            }
                        }
                        await broadcast({"type": "simulation_frame", "payload": frame_payload})
            except Exception as e:
                logging.error(f"Error generando frame de reinicio: {e}")
        
        msg = f"✅ Estado de simulación reiniciado (modo: {initial_mode})."
        if ws: await send_notification(ws, msg, "success")
        
    except Exception as e:
        logging.error(f"Error al reiniciar simulación: {e}", exc_info=True)
        if ws: await send_notification(ws, f"❌ Error al reiniciar: {str(e)}", "error")


async def handle_inject_energy(args):
    """Inyecta energía en el estado cuántico actual."""
    ws = g_state['websockets'].get(args.get('ws_id'))
    motor = g_state.get('motor')
    
    if not motor:
        if ws: await send_notification(ws, "⚠️ No hay modelo cargado.", "warning")
        return
    
    if hasattr(motor, 'native_engine'):
        if ws: await send_notification(ws, "⚠️ Inyección de energía no soportada aún en motor nativo.", "warning")
        return
        
    if not motor.state or motor.state.psi is None:
        if ws: await send_notification(ws, "⚠️ Estado no válido.", "warning")
        return
    
    energy_type = args.get('type', 'primordial_soup')
    
    try:
        psi = motor.state.psi
        if psi.dim() == 4: psi = psi[0]
        
        device = psi.device
        channels, height, width = psi.shape
        center_x, center_y = width // 2, height // 2
        psi_new = psi.clone()
        
        msg = ""
        if energy_type == 'primordial_soup':
            radius = min(20, width // 4)
            density = 0.3
            for x in range(max(0, center_x - radius), min(width, center_x + radius)):
                for y in range(max(0, center_y - radius), min(height, center_y + radius)):
                    dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
                    prob = density * torch.exp(torch.tensor(-dist / (radius/2), device=device))
                    if torch.rand(1, device=device).item() < prob.item():
                        noise = (torch.randn(channels, device=device) + 1j * torch.randn(channels, device=device)) * 0.1
                        psi_new[:, y, x] = psi_new[:, y, x] + noise
            msg = "🧪 Sopa Primordial inyectada"
            
        elif energy_type == 'dense_monolith':
            size = min(10, width // 8)
            intensity = 2.0
            for x in range(max(0, center_x - size), min(width, center_x + size)):
                for y in range(max(0, center_y - size), min(height, center_y + size)):
                    psi_new[:, y, x] = (torch.randn(channels, device=device) + 1j * torch.randn(channels, device=device)) * intensity * 0.1
            msg = "⬛ Monolito Denso inyectado"
            
        elif energy_type == 'symmetric_seed':
            size = min(8, width // 10)
            intensity = 1.5
            for x in range(center_x - size, center_x):
                for y in range(center_y - size, center_y):
                    base_state = (torch.randn(channels, device=device) + 1j * torch.randn(channels, device=device)) * intensity * 0.1
                    dx, dy = center_x - x, center_y - y
                    if 0 <= center_x + dx < width and 0 <= center_y + dy < height:
                        psi_new[:, center_y + dy, center_x + dx] = base_state
                    if 0 <= center_x - dx < width and 0 <= center_y + dy < height:
                        psi_new[:, center_y + dy, center_x - dx] = base_state
                    if 0 <= center_x + dx < width and 0 <= center_y - dy < height:
                        psi_new[:, center_y - dy, center_x + dx] = base_state
                    if 0 <= center_x - dx < width and 0 <= center_y - dy < height:
                        psi_new[:, center_y - dy, center_x - dx] = base_state
            msg = "🔬 Semilla Simétrica inyectada"
        
        if motor.state.psi.dim() == 4:
            motor.state.psi[0] = psi_new
        else:
            motor.state.psi = psi_new
            
        logging.info(msg)
        if ws: await send_notification(ws, msg, "success")
        
    except Exception as e:
        logging.error(f"Error inyectando energía: {e}", exc_info=True)
        if ws: await send_notification(ws, f"Error: {str(e)}", "error")





HANDLERS = {
    "play": handle_play,
    "pause": handle_pause,
    "load_experiment": handle_load_experiment,
    "unload_model": handle_unload_model,
    "switch_engine": handle_switch_engine,
    "reset": handle_reset,
    "inject_energy": handle_inject_energy,
    "set_inference_config": handle_set_inference_config
}
