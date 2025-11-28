import asyncio
import logging
import time
import numpy as np
import torch
from typing import Optional, Dict, Any
from .base_service import BaseService
from ..server.server_state import g_state
from ..server.data_compression import optimize_frame_payload
from ..pipelines.visualization_pipeline import VisualizationPipeline
from ..analysis.epoch_detector import EpochDetector

class DataProcessingService(BaseService):
    """
    Servicio responsable del procesamiento de datos y visualización.
    Consume estados de la simulación, genera visualizaciones y comprime datos.
    """
    
    def __init__(self, state_queue: asyncio.Queue, broadcast_queue: asyncio.Queue):
        super().__init__("DataProcessing")
        self.state_queue = state_queue
        self.broadcast_queue = broadcast_queue
        self.viz_pipeline = VisualizationPipeline()
        self.epoch_detector = EpochDetector()
        
    async def _start_impl(self):
        """Inicia el bucle de procesamiento."""
        self._task = asyncio.create_task(self._processing_loop())
        
    async def _stop_impl(self):
        """Detiene el bucle de procesamiento."""
        pass
        
    async def _processing_loop(self):
        """Bucle principal de procesamiento."""
        logging.info("🎨 DataProcessingService: Bucle iniciado.")
        
        while self._is_running:
            try:
                # 1. Obtener estado de la cola (con timeout para verificar _is_running)
                try:
                    state_data = await asyncio.wait_for(self.state_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                start_time = time.time()
                
                # 2. Extraer datos del estado
                motor = state_data['motor_ref']
                roi_manager = state_data['roi']
                viz_type = state_data['viz_type']
                step = state_data['step']
                
                # 3. Obtener estado denso (Heavy Operation)
                # Offload to thread pool
                roi = None
                if roi_manager and roi_manager.roi_enabled:
                    roi = (
                        roi_manager.roi_x,
                        roi_manager.roi_y,
                        roi_manager.roi_x + roi_manager.roi_width,
                        roi_manager.roi_y + roi_manager.roi_height
                    )
                
                # Callback para verificar pausa durante conversión (opcional, ya que este servicio puede tardar)
                def check_pause():
                    return g_state.get('is_paused', True)
                
                # Obtener estado denso según tipo de motor
                # Motor nativo tiene get_dense_state() con lazy conversion
                # Motor Python tiene state.psi directamente
                if hasattr(motor, 'get_dense_state'):
                    # Motor nativo: usar get_dense_state() con ROI y verificación de pausa
                    psi = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: motor.get_dense_state(roi=roi, check_pause_callback=check_pause)
                    )
                else:
                    # Motor Python: acceder directamente a state.psi
                    psi = motor.state.psi
                
                if psi is None:
                    self.state_queue.task_done()
                    continue

                # --- ANÁLISIS DE ÉPOCA ---
                # Ejecutar análisis en thread pool para no bloquear
                epoch_metrics = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.epoch_detector.analyze_state,
                    psi
                )
                current_epoch = self.epoch_detector.determine_epoch(epoch_metrics)
                
                # Actualizar estado global
                g_state['current_epoch'] = current_epoch
                g_state['epoch_metrics'] = epoch_metrics
                
                # 4. Generar visualización (Heavy Operation)
                # Offload to thread pool
                viz_data = await self.viz_pipeline.generate_frame(
                    psi, 
                    step,
                    viz_type=viz_type
                )
                
                if not viz_data:
                    self.state_queue.task_done()
                    continue
                
                # 5. Comprimir y optimizar payload (Heavy Operation)
                # optimize_frame_payload ya usa run_in_executor internamente para compresión
                final_payload = await optimize_frame_payload(
                    viz_data,
                    enable_compression=True, # Siempre comprimir para eficiencia
                    downsample_factor=1, # Configurable si es necesario
                    viz_type=viz_type
                )
                
                # Agregar metadatos de simulación
                final_payload['simulation_info'] = {
                    'step': step,
                    'is_paused': g_state.get('is_paused', False),
                    'live_feed_enabled': True,
                    'fps': g_state.get('current_fps', 0.0),
                    'epoch': current_epoch,
                    'epoch_metrics': epoch_metrics
                }
                
                # 6. Enviar a cola de broadcast
                if not self.broadcast_queue.full():
                    try:
                        self.broadcast_queue.put_nowait({
                            'type': 'simulation_frame',
                            'payload': final_payload
                        })
                    except asyncio.QueueFull:
                        pass
                
                self.state_queue.task_done()
                
                # Yield para no acaparar el loop
                await asyncio.sleep(0)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"❌ Error en bucle de procesamiento: {e}")
                await asyncio.sleep(0.1)
