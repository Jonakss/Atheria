"""
Wrapper para integrar el motor nativo de C++ (atheria_core.Engine) con el frontend.

Este wrapper proporciona una interfaz compatible con Aetheria_Motor para que
el motor nativo de alto rendimiento pueda usarse como reemplazo directo.
"""
import torch
import logging
import numpy as np
from typing import Optional

# Versión del wrapper
try:
    from .__version__ import __version__ as ENGINE_VERSION
except ImportError:
    ENGINE_VERSION = "4.1.0"  # Fallback

# Intentar importar el módulo nativo con manejo robusto de errores CUDA
NATIVE_AVAILABLE = False
_native_import_error = None
_native_cuda_issue = False  # Flag para indicar que hay problema de CUDA pero el módulo existe

try:
    # Intentar importar el módulo nativo
    import atheria_core
    NATIVE_AVAILABLE = True
    logging.info("✅ Módulo nativo atheria_core importado exitosamente")
except (ImportError, OSError, RuntimeError) as e:
    error_str = str(e)
    _native_import_error = error_str
    
    # Detectar problemas específicos de CUDA runtime
    cuda_runtime_keywords = [
        '__nvJitLinkCreate',
        'libnvJitLink',
        'libcusparse.so',
        'undefined symbol'
    ]
    
    if any(keyword in error_str for keyword in cuda_runtime_keywords):
        # Problema de CUDA runtime - el módulo está compilado pero tiene problemas de CUDA
        _native_cuda_issue = True
        logging.warning(f"⚠️ Problema de CUDA runtime detectado al importar atheria_core: {error_str[:100]}")
        logging.info("💡 El motor nativo solo funcionará en modo CPU. Usa device='cpu' al inicializar.")
        # No marcamos como no disponible - aún puede funcionar en CPU
        # NATIVE_AVAILABLE permanece False, pero el wrapper puede intentar inicializar en CPU
    else:
        # Otro tipo de error - probablemente el módulo no está compilado
        logging.warning(f"atheria_core no disponible: {error_str[:100]}")
        if "No module named" in error_str or "cannot open shared object file" in error_str:
            logging.info("💡 El módulo C++ no está compilado. Ejecuta: python setup.py build_ext --inplace")
        else:
            logging.info("💡 Error inesperado al importar módulo nativo. Usando motor Python como fallback.")
except Exception as e:
    # Error inesperado
    _native_import_error = str(e)
    logging.warning(f"Error inesperado importando atheria_core: {e}")
    logging.info("El motor nativo no estará disponible, usando motor Python como fallback.")

from ..engines.qca_engine import QuantumState


class NativeEngineWrapper:
    """
    Wrapper que envuelve atheria_core.Engine para mantener compatibilidad
    con el código existente que usa Aetheria_Motor.
    
    Convierte entre el formato disperso del motor nativo y el formato
    denso (grid) usado por el frontend.
    """
    
    # Versión del wrapper Python
    VERSION = ENGINE_VERSION
    
    def __init__(self, grid_size: int, d_state: int, device: str = None, cfg=None):
        """
        Inicializa el wrapper del motor nativo.
        
        Args:
            grid_size: Tamaño del grid para visualización (debe coincidir con inference_grid_size)
            d_state: Dimensión del estado cuántico
            device: Dispositivo ('cpu', 'cuda', o None para auto-detección)
            cfg: Configuración del experimento (opcional)
        """
        # Si device es None, usar auto-detección desde config
        if device is None:
            from src import config as global_cfg
            device = global_cfg.get_native_device()
        # Verificar disponibilidad del módulo
        # Intentar importar el módulo si no está disponible pero hay problema de CUDA
        if not NATIVE_AVAILABLE and _native_cuda_issue and device == "cpu":
            logging.info("Intentando importar módulo nativo forzando CPU mode...")
            try:
                import os
                # Forzar CPU deshabilitando CUDA temporalmente
                original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                try:
                    # Reimportar el módulo si ya se importó antes
                    import sys
                    if 'atheria_core' in sys.modules:
                        del sys.modules['atheria_core']
                    import atheria_core  # Reintentar importación
                    # Importar exitoso - no necesitamos hacer nada más
                    logging.info("✅ Módulo nativo importado exitosamente en CPU mode")
                finally:
                    # Restaurar valor original
                    if original_cuda is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
                    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                        del os.environ['CUDA_VISIBLE_DEVICES']
                
                # Verificar que ahora está disponible
                import atheria_core
            except Exception as e2:
                # Aún falla - no disponible
                error_msg = f"atheria_core no está disponible. Error original: {_native_import_error[:100] if _native_import_error else str(e2)}"
                if _native_cuda_issue:
                    error_msg += " (Problema de CUDA runtime - solo CPU mode disponible, pero también falló)"
                raise ImportError(error_msg + " Usa el motor Python como fallback.")
        
        # Verificar una vez más después del intento - asegurar que atheria_core esté disponible
        # Importar atheria_core para usarlo en el resto del método
        import atheria_core as atheria_core_module
        
        # Verificar una vez más si no estaba disponible antes
        if not NATIVE_AVAILABLE:
            # Intentar verificar que funciona
            try:
                # Probar crear un Engine temporal para verificar
                test_engine = atheria_core_module.Engine(d_state=1, device='cpu')
                del test_engine
            except (ImportError, OSError, RuntimeError) as e:
                # No disponible para nada
                error_msg = "atheria_core no está disponible."
                if _native_import_error:
                    error_msg += f" Error: {_native_import_error[:100]}"
                elif str(e):
                    error_msg += f" Error: {str(e)[:100]}"
                if _native_cuda_issue:
                    error_msg += " (Problema de CUDA runtime - intenta usar device='cpu')"
                raise ImportError(error_msg + " Usa el motor Python como fallback.")
        
        # Si hay problema de CUDA runtime, verificar si se puede usar CPU mode
        if _native_cuda_issue:
            if device == "cuda":
                logging.warning("⚠️ Problema de CUDA runtime detectado al importar atheria_core.")
                logging.info("💡 Intentando usar CPU mode para motor nativo...")
                # Intentar CPU mode en lugar de fallar
                device = "cpu"
                # Asegurar que CUDA_VISIBLE_DEVICES esté deshabilitado para evitar conflictos
                import os
                original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                try:
                    # Verificar que el motor funciona en CPU mode
                    test_engine = atheria_core_module.Engine(d_state=1, device='cpu')
                    del test_engine
                    logging.info("✅ Motor nativo funciona en CPU mode")
                except Exception as e:
                    logging.error(f"❌ Motor nativo también falla en CPU mode: {e}")
                    raise ImportError(f"Motor nativo no disponible (ni CUDA ni CPU): {e}")
                finally:
                    # Restaurar CUDA_VISIBLE_DEVICES
                    if original_cuda_visible is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                        del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Verificar que device es válido antes de inicializar
        if device not in ('cpu', 'cuda'):
            logging.warning(f"⚠️ Device '{device}' no válido. Usando 'cpu'.")
            device = 'cpu'
        
        self.grid_size = int(grid_size)
        self.d_state = int(d_state)
        self.device_str = device
        self.device = torch.device(device)
        self.cfg = cfg
        
        # Intentar inicializar motor nativo con manejo robusto de errores
        try:
            # Inicializar motor nativo con el tamaño del grid
            # El grid_size se usa para construir inputs del modelo con el tamaño correcto
            self.native_engine = atheria_core_module.Engine(d_state=d_state, device=device, grid_size=grid_size)
            
            # Obtener versión del motor nativo C++ después de inicializarlo
            try:
                if hasattr(atheria_core_module, 'get_version'):
                    self.native_version = atheria_core_module.get_version()
                elif hasattr(atheria_core_module, '__version__'):
                    self.native_version = atheria_core_module.__version__
                else:
                    self.native_version = "unknown"
            except Exception as e:
                logging.debug(f"No se pudo obtener versión del motor nativo: {e}")
                self.native_version = "unknown"
            
            logging.info(f"✅ Motor nativo C++ inicializado (device={device}, grid_size={grid_size}, version={self.native_version})")
        except RuntimeError as e:
            error_str = str(e)
            if device == 'cuda' and ('cuda' in error_str.lower() or '101' in error_str):
                logging.warning(f"⚠️ Error inicializando motor nativo en CUDA: {e}")
                logging.info("💡 Intentando fallback a CPU...")
                # Intentar CPU mode como fallback
                device = 'cpu'
                self.device_str = device
                self.device = torch.device(device)
                self.native_engine = atheria_core_module.Engine(d_state=d_state, device='cpu', grid_size=grid_size)
                
                # Obtener versión del motor nativo C++ después de inicializarlo
                try:
                    if hasattr(atheria_core_module, 'get_version'):
                        self.native_version = atheria_core_module.get_version()
                    elif hasattr(atheria_core_module, '__version__'):
                        self.native_version = atheria_core_module.__version__
                    else:
                        self.native_version = "unknown"
                except Exception as e:
                    logging.debug(f"No se pudo obtener versión del motor nativo: {e}")
                    self.native_version = "unknown"
                
                logging.info(f"✅ Motor nativo inicializado en CPU mode (fallback desde CUDA, version={self.native_version})")
            else:
                raise
        
        # Estado cuántico para compatibilidad (denso)
        # El motor nativo usa formato disperso, pero necesitamos denso para visualización
        self.state = QuantumState(grid_size, d_state, device)
        
        # Configuración
        self.grid_size = grid_size
        self.d_state = d_state
        
        # Estado interno
        self.model_loaded = False
        self.step_count = 0
        
        # Para compatibilidad con código existente
        self.operator = None  # El modelo está cargado en el motor nativo
        self.original_model = None
        self.optimizer = None
        self.has_memory = False
        self.is_compiled = False
        
        # Para visualizaciones (delta_psi)
        self.last_delta_psi = None
        self.last_psi_input = None
        self.last_delta_psi_decay = None
        
        # OPTIMIZACIÓN: Lazy conversion - solo convertir cuando se necesita
        self._dense_state_stale = True  # Estado denso está desactualizado después de evolve
        self._last_conversion_roi = None  # Última ROI usada para conversión
        
        logging.info(f"NativeEngineWrapper inicializado (grid_size={grid_size}, d_state={d_state}, device={device})")
    
    def load_model(self, model_path: str) -> bool:
        """
        Carga un modelo TorchScript (.pt) en el motor nativo.
        
        Args:
            model_path: Ruta al archivo .pt de TorchScript
            
        Returns:
            True si se cargó exitosamente
        """
        success = self.native_engine.load_model(model_path)
        self.model_loaded = success
        if success:
            logging.info(f"✅ Modelo TorchScript cargado en motor nativo: {model_path}")
        else:
            # Obtener mensaje de error específico desde C++
            error_msg = self.native_engine.get_last_error()
            if error_msg:
                logging.error(f"❌ Error al cargar modelo TorchScript: {model_path}")
                logging.error(f"   Detalle del error: {error_msg}")
            else:
                logging.error(f"❌ Error al cargar modelo TorchScript: {model_path} (error desconocido)")
        return success
    
    def evolve_internal_state(self):
        """Evoluciona el estado interno usando el motor nativo."""
        if not self.model_loaded:
            logging.warning("Modelo no cargado. No se puede evolucionar el estado.")
            return
        
        # Ejecutar paso nativo (todo en C++)
        particle_count = self.native_engine.step_native()
        self.step_count += 1
        
        # OPTIMIZACIÓN CRÍTICA: NO convertir aquí - solo marcar como "stale"
        # La conversión se hará solo cuando se necesite (lazy conversion)
        # Esto evita convertir 65,536 coordenadas en cada paso
        self._dense_state_stale = True
    
    def get_dense_state(self, roi=None, check_pause_callback=None):
        """
        Obtiene el estado denso, convirtiendo solo si es necesario.
        
        Args:
            roi: Region of Interest (x_min, y_min, x_max, y_max) opcional. Si se proporciona,
                 solo convierte esa región. Si None, convierte todo el grid.
            check_pause_callback: Función callback opcional para verificar pausa.
                                 Debe retornar True si está pausado.
        
        Returns:
            Tensor complejo con el estado denso [1, H, W, d_state]
        """
        # Convertir solo si el estado está desactualizado o no existe
        # O si la ROI cambió (necesitamos convertir más/menos del grid)
        roi_changed = (roi is not None and roi != self._last_conversion_roi) or \
                      (roi is None and self._last_conversion_roi is not None)
        
        if self._dense_state_stale or self.state.psi is None or roi_changed:
            self._update_dense_state_from_sparse(roi=roi, check_pause_callback=check_pause_callback)
            self._dense_state_stale = False
            self._last_conversion_roi = roi
        
        return self.state.psi
    
    def _update_dense_state_from_sparse(self, roi=None, check_pause_callback=None):
        """
        Convierte el estado disperso del motor nativo a formato denso (grid)
        para compatibilidad con el frontend.
        
        OPTIMIZACIONES CRÍTICAS:
        - Lazy conversion: solo se llama cuando se necesita
        - ROI support: solo convierte región visible si se proporciona
        - Pause check: verifica pausa periódicamente durante conversión
        
        Args:
            roi: Region of Interest (x_min, y_min, x_max, y_max) opcional.
                 Si se proporciona, solo convierte esa región.
            check_pause_callback: Función callback para verificar pausa.
                                 Debe retornar True si está pausado.
        """
        # Inicializar grid denso si no existe
        if self.state.psi is None:
            self.state.psi = torch.zeros(
                1, self.grid_size, self.grid_size, self.d_state,
                dtype=torch.complex64, device=self.device
            )
        
        # OPTIMIZACIÓN: Determinar región a convertir (ROI o todo el grid)
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            # Asegurar que ROI esté dentro del grid
            x_min = max(0, min(x_min, self.grid_size))
            y_min = max(0, min(y_min, self.grid_size))
            x_max = max(x_min, min(x_max, self.grid_size))
            y_max = max(y_min, min(y_max, self.grid_size))
            
            # Crear lista de coordenadas solo para ROI
            coords_list = [
                atheria_core.Coord3D(x, y, 0)
                for y in range(y_min, y_max)
                for x in range(x_min, x_max)
            ]
            logging.debug(f"Convirtiendo solo ROI: ({x_min}, {y_min}) - ({x_max}, {y_max}), {len(coords_list)} coordenadas")
        else:
            # Sin ROI: convertir todo el grid (fallback)
            coords_list = [
                atheria_core.Coord3D(x, y, 0)
                for y in range(self.grid_size)
                for x in range(self.grid_size)
            ]
            logging.debug(f"Convirtiendo todo el grid: {len(coords_list)} coordenadas")
        
        # OPTIMIZACIÓN: Usar batching y verificación de pausa
        try:
            # Obtener lista de coordenadas activas del motor nativo si está disponible
            # Si no está disponible, usar la lista completa (ROI o todo el grid)
            active_coords = None
            try:
                # Intentar obtener coordenadas activas (si el motor nativo lo expone)
                if hasattr(self.native_engine, 'get_active_coords'):
                    active_coords_list = self.native_engine.get_active_coords()
                    if active_coords_list and len(active_coords_list) > 0:
                        # Filtrar coordenadas activas por ROI si aplica
                        if roi is not None:
                            x_min, y_min, x_max, y_max = roi
                            active_coords = [
                                coord for coord in active_coords_list
                                if x_min <= coord.x < x_max and y_min <= coord.y < y_max
                            ]
                        else:
                            active_coords = active_coords_list
            except:
                pass
            
            # Si tenemos coordenadas activas, usar solo esas (mucho más rápido)
            if active_coords is not None and len(active_coords) > 0:
                coords_to_process = active_coords
                BATCH_SIZE = 1000  # Batch más grande para coordenadas activas
                logging.debug(f"Usando {len(active_coords)} coordenadas activas (ROI aplicado)")
            else:
                # Sin coordenadas activas: usar lista completa (ROI o todo el grid)
                coords_to_process = coords_list
                BATCH_SIZE = 500  # Batch más pequeño para lista completa
                logging.debug(f"Usando lista completa: {len(coords_to_process)} coordenadas")
            
            # Procesar en batches con verificación de pausa
            for i in range(0, len(coords_to_process), BATCH_SIZE):
                # CRÍTICO: Verificar pausa cada batch para permitir pausa inmediata
                if check_pause_callback and check_pause_callback():
                    logging.debug("Conversión interrumpida por pausa")
                    return  # Salir temprano si está pausado
                
                batch_coords = coords_to_process[i:i+BATCH_SIZE]
                for coord in batch_coords:
                    try:
                        state_tensor = self.native_engine.get_state_at(coord)
                        
                        # Verificar shape antes de copiar
                        if state_tensor.shape == (self.d_state,):
                            # Mover a dispositivo correcto si es necesario
                            if state_tensor.device != self.device:
                                state_tensor = state_tensor.to(self.device)
                            
                            # Copiar al estado denso (batch, H, W, d_state)
                            if 0 <= coord.y < self.grid_size and 0 <= coord.x < self.grid_size:
                                self.state.psi[0, coord.y, coord.x] = state_tensor
                    except Exception as e:
                        logging.debug(f"Error obteniendo estado en ({coord.x}, {coord.y}): {e}")
                        
        except Exception as e:
            logging.warning(f"Error convirtiendo estado disperso a denso: {e}")
            # En caso de error, mantener grid actual o inicializar vacío
            if self.state.psi is None:
                self.state.psi = torch.zeros(
                    1, self.grid_size, self.grid_size, self.d_state,
                    dtype=torch.complex64, device=self.device
                )
    
    def get_model_for_params(self):
        """Retorna el modelo para acceso a parámetros (compatibilidad)."""
        return self.original_model if self.original_model else self
    
    def compile_model(self):
        """Método de compatibilidad - el modelo nativo ya está compilado."""
        self.is_compiled = True
        logging.info("Modelo nativo: ya está optimizado (compilado en C++)")
    
    def add_initial_particles(self, num_particles: int = 10):
        """
        Agrega partículas iniciales aleatorias al motor nativo.
        
        Args:
            num_particles: Número de partículas a agregar
        """
        # Generar partículas aleatorias en el grid
        for _ in range(num_particles):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            z = 0  # Para 2D, z=0
            
            # Estado inicial pequeño (energía baja)
            initial_state = torch.randn(self.d_state, dtype=torch.complex64) * 0.1
            
            # Agregar al motor nativo
            coord = atheria_core.Coord3D(x, y, z)
            self.native_engine.add_particle(coord, initial_state)
        
        logging.info(f"✅ {num_particles} partículas iniciales agregadas al motor nativo")
        
        # Actualizar estado denso (solo si se necesita)
        # Usar get_dense_state() para lazy conversion
        self.get_dense_state()
    
    def cleanup(self):
        """
        Limpia recursos del motor nativo de forma explícita.
        Debe llamarse antes de destruir el wrapper para evitar segfaults.
        """
        try:
            # Limpiar estado denso primero
            if hasattr(self, 'state') and self.state is not None:
                if hasattr(self.state, 'psi') and self.state.psi is not None:
                    self.state.psi = None
                self.state = None
            
            # Limpiar referencias al motor nativo
            # El destructor de C++ se llamará automáticamente cuando no haya más referencias
            if hasattr(self, 'native_engine') and self.native_engine is not None:
                # Liberar referencias explícitamente
                self.native_engine = None
            
            # Limpiar otras referencias
            self.model_loaded = False
            self.step_count = 0
            self.last_delta_psi = None
            self.last_psi_input = None
            self.last_delta_psi_decay = None
            
            logging.debug("NativeEngineWrapper limpiado correctamente")
        except Exception as e:
            logging.warning(f"Error durante cleanup de NativeEngineWrapper: {e}")
    
    def __del__(self):
        """Destructor - llama a cleanup para asegurar limpieza correcta."""
        try:
            self.cleanup()
        except Exception:
            # Ignorar errores en destructor para evitar problemas durante garbage collection
            pass
    
    @property
    def _state_psi_property(self):
        """
        Property helper para acceder a state.psi con lazy conversion.
        Usado internamente para compatibilidad con código existente.
        """
        return self.get_dense_state(check_pause_callback=lambda: False)
    
    def _ensure_state_psi(self):
        """
        Asegura que state.psi esté disponible (para compatibilidad).
        Internamente usa lazy conversion.
        """
        if self.state.psi is None or self._dense_state_stale:
            self.get_dense_state(check_pause_callback=lambda: False)

