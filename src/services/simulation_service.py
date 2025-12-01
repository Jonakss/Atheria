import asyncio
import logging
import time
import torch
from typing import Optional, Dict, Any
from .base_service import BaseService
from ..server.server_state import g_state

class SimulationService(BaseService):
    """
    Servicio responsable de ejecutar el bucle de simulación física.
    Desacoplado de la visualización y el procesamiento de datos.
    """
    
    def __init__(self, state_queue: asyncio.Queue):
        super().__init__("Simulation")
        self.state_queue = state_queue
        self.motor = None
        self.target_fps = 60.0
        self.frame_time = 1.0 / self.target_fps
        
        # FPS Calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_update_interval = 0.5 # Update every 0.5 seconds
        
    async def _start_impl(self):
        """Inicia el bucle de simulación."""
        self._task = asyncio.create_task(self._simulation_loop())
        
    async def _stop_impl(self):
        """Detiene el bucle de simulación."""
        # El task se cancela en la clase base
        pass
        
    async def _simulation_loop(self):
        """Bucle principal de simulación."""
        logging.info("🌀 SimulationService: Bucle iniciado.")
        
        last_time = time.time()
        
        while self._is_running:
            try:
                start_time = time.time()
                
                # 1. Verificar pausa
                if g_state.get('is_paused', True):
                    await asyncio.sleep(0.1)
                    continue
                
                # 2. Obtener motor actual
                self.motor = g_state.get('motor')
                if not self.motor:
                    await asyncio.sleep(0.1)
                    continue
                
                # 3. Ejecutar paso de física (Physics Step)
                # Offload to thread pool to avoid blocking event loop
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"🔄 Ejecutando paso de física (Step {g_state.get('simulation_step', 0)})...")
                
                await asyncio.get_event_loop().run_in_executor(None, self.motor.evolve_internal_state)
                
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("✅ Paso de física completado.")
                
                # 4. Actualizar contadores
                current_step = g_state.get('simulation_step', 0)
                g_state['simulation_step'] = current_step + 1
                
                # FPS Calculation
                self.fps_counter += 1
                current_time = time.time()
                time_diff = current_time - self.fps_start_time
                
                if time_diff >= self.fps_update_interval:
                    current_fps = self.fps_counter / time_diff
                    g_state['current_fps'] = current_fps
                    
                    # Reset counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                    
                    # Log FPS occasionally
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"📊 FPS: {current_fps:.1f}")
                
                # 5. Enviar estado a cola de procesamiento (si hay espacio)
                # Solo enviamos referencias o datos ligeros, el procesamiento pesado lo hace DataProcessingService
                if not self.state_queue.full():
                    # Para motor nativo, pasamos referencia al motor y ROI
                    # Para motor python, pasamos el estado psi
                    state_data = {
                        'step': current_step + 1,
                        'timestamp': time.time(),
                        'motor_ref': self.motor, # Referencia para que DataProcessing extraiga datos
                        'roi': g_state.get('roi_manager'), # Info de ROI actual
                        'viz_type': g_state.get('viz_type', 'density')
                    }
                    
                    # Usar put_nowait para no bloquear el loop de física
                    try:
                        self.state_queue.put_nowait(state_data)
                    except asyncio.QueueFull:
                        pass # Si la cola está llena, descartamos frame (frame skipping)
                
                # 6. Control de FPS
                elapsed = time.time() - start_time
                wait_time = max(0, self.frame_time - elapsed)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(0) # Yield para permitir otras tareas
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"❌ Error en bucle de simulación: {e}")
                await asyncio.sleep(1.0) # Esperar antes de reintentar
