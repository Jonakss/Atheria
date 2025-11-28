"""
Wrapper para integrar el motor nativo de C++ (atheria_core.Engine) con el frontend.

Este wrapper proporciona una interfaz compatible con Aetheria_Motor para que
el motor nativo de alto rendimiento pueda usarse como reemplazo directo.
"""

import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Optional
from pathlib import Path
import sys

# Versión del wrapper
try:
    from .__version__ import __version__ as ENGINE_VERSION
except ImportError:
    ENGINE_VERSION = "4.1.0"  # Fallback

# Intentar importar el módulo nativo con manejo robusto de errores CUDA
NATIVE_AVAILABLE = False
_native_import_error = None
_native_cuda_issue = (
    False  # Flag para indicar que hay problema de CUDA pero el módulo existe
)

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
        "__nvJitLinkCreate",
        "libnvJitLink",
        "libcusparse.so",
        "undefined symbol",
    ]

    if any(keyword in error_str for keyword in cuda_runtime_keywords):
        # Problema de CUDA runtime - el módulo está compilado pero tiene problemas de CUDA
        _native_cuda_issue = True
        logging.warning(
            f"⚠️ Problema de CUDA runtime detectado al importar atheria_core: {error_str[:100]}"
        )
        logging.info(
            "💡 El motor nativo solo funcionará en modo CPU. Usa device='cpu' al inicializar."
        )
        # No marcamos como no disponible - aún puede funcionar en CPU
        # NATIVE_AVAILABLE permanece False, pero el wrapper puede intentar inicializar en CPU
    else:
        # Otro tipo de error - probablemente el módulo no está compilado
        logging.warning(f"atheria_core no disponible: {error_str[:100]}")
        if (
            "No module named" in error_str
            or "cannot open shared object file" in error_str
        ):
            logging.info(
                "💡 El módulo C++ no está compilado. Ejecuta: python setup.py build_ext --inplace"
            )
        else:
            logging.info(
                "💡 Error inesperado al importar módulo nativo. Usando motor Python como fallback."
            )
except Exception as e:
    # Error inesperado
    _native_import_error = str(e)
    logging.warning(f"Error inesperado importando atheria_core: {e}")
    logging.info(
        "El motor nativo no estará disponible, usando motor Python como fallback."
    )

from ..engines.qca_engine import QuantumState


def export_model_to_jit(
    model: nn.Module,
    experiment_name: str,
    example_input_shape: tuple,
    output_dir: Optional[str] = None,
) -> str:
    """
    Exporta un modelo PyTorch a formato TorchScript (JIT).

    Args:
        model: Instancia del modelo PyTorch a exportar.
        experiment_name: Nombre del experimento, usado para generar el nombre del archivo.
        example_input_shape: Shape del tensor de entrada de ejemplo (ej: (1, 8, 256, 256)).
        output_dir: Directorio de salida opcional. Si es None, usa el directorio de checkpoints.

    Returns:
        Ruta al archivo .pt exportado.
    """
    from .. import config as global_cfg
    import os

    if output_dir is None:
        output_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, experiment_name)

    os.makedirs(output_dir, exist_ok=True)

    # Usar un timestamp para nombre de archivo único para evitar problemas de caché
    timestamp = int(torch.randint(0, 100000, (1,)).item())
    output_path = os.path.join(output_dir, f"model_jit_{timestamp}.pt")

    try:
        logging.info(f"📦 Exportando modelo a TorchScript: {output_path}")
        model.eval()
        device = next(model.parameters()).device
        example_input = torch.randn(example_input_shape, device=device)

        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input, strict=False)

        traced_model.save(output_path)
        logging.info(f"✅ Modelo exportado exitosamente a: {output_path}")

        # Verificación
        logging.info("   Verificando modelo exportado...")
        loaded_model = torch.jit.load(output_path, map_location=device)
        test_output = loaded_model(example_input)
        logging.info(f"✅ Modelo verificado. Salida de prueba: {test_output.shape}")

        return output_path

    except Exception as e:
        logging.error(f"❌ Error al exportar modelo a JIT: {e}", exc_info=True)
        raise


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
                original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    # Reimportar el módulo si ya se importó antes
                    import sys

                    if "atheria_core" in sys.modules:
                        del sys.modules["atheria_core"]
                    import atheria_core  # Reintentar importación

                    # Importar exitoso - no necesitamos hacer nada más
                    logging.info("✅ Módulo nativo importado exitosamente en CPU mode")
                finally:
                    # Restaurar valor original
                    if original_cuda is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
                    elif "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]

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
                test_engine = atheria_core_module.Engine(d_state=1, device="cpu")
                del test_engine
            except (ImportError, OSError, RuntimeError) as e:
                # No disponible para nada
                error_msg = "atheria_core no está disponible."
                if _native_import_error:
                    error_msg += f" Error: {_native_import_error[:100]}"
                elif str(e):
                    error_msg += f" Error: {str(e)[:100]}"
                if _native_cuda_issue:
                    error_msg += (
                        " (Problema de CUDA runtime - intenta usar device='cpu')"
                    )
                raise ImportError(error_msg + " Usa el motor Python como fallback.")

        # Si hay problema de CUDA runtime, verificar si se puede usar CPU mode
        if _native_cuda_issue:
            if device == "cuda":
                logging.warning(
                    "⚠️ Problema de CUDA runtime detectado al importar atheria_core."
                )
                logging.info("💡 Intentando usar CPU mode para motor nativo...")
                # Intentar CPU mode en lugar de fallar
                device = "cpu"
                # Asegurar que CUDA_VISIBLE_DEVICES esté deshabilitado para evitar conflictos
                import os

                original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    # Verificar que el motor funciona en CPU mode
                    test_engine = atheria_core_module.Engine(d_state=1, device="cpu")
                    del test_engine
                    logging.info("✅ Motor nativo funciona en CPU mode")
                except Exception as e:
                    logging.error(f"❌ Motor nativo también falla en CPU mode: {e}")
                    raise ImportError(
                        f"Motor nativo no disponible (ni CUDA ni CPU): {e}"
                    )
                finally:
                    # Restaurar CUDA_VISIBLE_DEVICES
                    if original_cuda_visible is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                    elif "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]

        # Verificar que device es válido antes de inicializar
        if device not in ("cpu", "cuda"):
            logging.warning(f"⚠️ Device '{device}' no válido. Usando 'cpu'.")
            device = "cpu"

        self.grid_size = int(grid_size)
        self.d_state = int(d_state)
        self.device_str = device
        self.device = torch.device(device)
        self.cfg = cfg

        # Intentar inicializar motor nativo con manejo robusto de errores
        try:
            # Inicializar motor nativo con el tamaño del grid
            # El grid_size se usa para construir inputs del modelo con el tamaño correcto
            self.native_engine = atheria_core_module.Engine(
                d_state=d_state, device=device, grid_size=grid_size
            )

            # Obtener versión del motor nativo C++ después de inicializarlo
            try:
                if hasattr(atheria_core_module, "get_version"):
                    self.native_version = atheria_core_module.get_version()
                elif hasattr(atheria_core_module, "__version__"):
                    self.native_version = atheria_core_module.__version__
                else:
                    self.native_version = "unknown"
            except Exception as e:
                logging.debug(f"No se pudo obtener versión del motor nativo: {e}")
                self.native_version = "unknown"

            logging.info(
                f"✅ Motor nativo C++ inicializado (device={device}, grid_size={grid_size}, version={self.native_version})"
            )
        except RuntimeError as e:
            error_str = str(e)
            if device == "cuda" and ("cuda" in error_str.lower() or "101" in error_str):
                logging.warning(f"⚠️ Error inicializando motor nativo en CUDA: {e}")
                logging.info("💡 Intentando fallback a CPU...")
                # Intentar CPU mode como fallback
                device = "cpu"
                self.device_str = device
                self.device = torch.device(device)
                self.native_engine = atheria_core_module.Engine(
                    d_state=d_state, device="cpu", grid_size=grid_size
                )

                # Obtener versión del motor nativo C++ después de inicializarlo
                try:
                    if hasattr(atheria_core_module, "get_version"):
                        self.native_version = atheria_core_module.get_version()
                    elif hasattr(atheria_core_module, "__version__"):
                        self.native_version = atheria_core_module.__version__
                    else:
                        self.native_version = "unknown"
                except Exception as e:
                    logging.debug(f"No se pudo obtener versión del motor nativo: {e}")
                    self.native_version = "unknown"

                logging.info(
                    f"✅ Motor nativo inicializado en CPU mode (fallback desde CUDA, version={self.native_version})"
                )
            else:
                raise

        # CRÍTICO: Generar estado inicial según INITIAL_STATE_MODE_INFERENCE (como motor Python)
        # El motor nativo usa formato disperso, pero generamos estado denso primero
        # y luego lo convertimos al formato disperso del motor nativo
        initial_mode = "complex_noise"  # Default
        if cfg is not None:
            initial_mode = getattr(cfg, "INITIAL_STATE_MODE_INFERENCE", "complex_noise")

        # Soporte para grid scaling: si training_grid_size < inference_grid_size, replicar estado
        base_state = None
        base_grid_size = None
        if cfg is not None:
            training_grid_size = getattr(cfg, "GRID_SIZE_TRAINING", None)
            if training_grid_size and training_grid_size < grid_size:
                # Crear estado base del tamaño de entrenamiento y replicarlo
                base_state_temp = QuantumState(
                    training_grid_size, d_state, device, initial_mode=initial_mode
                )
                base_state = base_state_temp.psi
                base_grid_size = training_grid_size
                logging.info(
                    f"🔄 Grid escalado: Creando estado base {training_grid_size}x{training_grid_size} para replicar en {grid_size}x{grid_size}"
                )

        # Estado cuántico para compatibilidad (denso)
        # El motor nativo usa formato disperso, pero necesitamos denso para visualización
        self.state = QuantumState(
            grid_size,
            d_state,
            device,
            initial_mode=initial_mode,
            base_state=base_state,
            base_grid_size=base_grid_size,
        )

        # CRÍTICO: Convertir estado denso inicial al formato disperso del motor nativo
        # Esto genera las partículas iniciales desde el estado denso (ley M respetada)
        self._initialize_native_state_from_dense(self.state.psi)

        # Configuración
        self.grid_size = grid_size
        self.d_state = d_state
        self.cfg = cfg  # Guardar cfg para regenerar estado inicial si es necesario

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
        self._dense_state_stale = (
            True  # Estado denso está desactualizado después de evolve
        )
        self._last_conversion_roi = None  # Última ROI usada para conversión

        logging.info(
            f"NativeEngineWrapper inicializado (grid_size={grid_size}, d_state={d_state}, device={device}, initial_mode={initial_mode})"
        )

    def _initialize_native_state_from_dense(self, dense_psi: torch.Tensor):
        """
        Convierte estado denso inicial al formato disperso del motor nativo.

        Esto respeta INITIAL_STATE_MODE_INFERENCE y genera partículas desde el estado denso,
        en lugar de agregar partículas manualmente. Las partículas emergen del estado inicial
        generado según la ley M (modelo cuántico).

        Args:
            dense_psi: Tensor complejo [1, H, W, d_state] con estado inicial denso
        """
        if dense_psi is None or dense_psi.numel() == 0:
            logging.warning(
                "⚠️ Estado denso inicial vacío. No se inicializará motor nativo."
            )
            return

        # Obtener valores absolutos para determinar qué células tienen estado significativo
        psi_abs = dense_psi.abs()
        psi_abs_sq = psi_abs.pow(2)

        # Umbral para considerar una célula como "activa" (tiene partícula)
        # Usar un umbral dinámico basado en la distribución de valores
        # CRÍTICO: complex_noise genera valores pequeños (~0.1), necesitamos umbral más permisivo
        psi_abs_sq_max = psi_abs_sq.max().item()

        # Para complex_noise, valores típicos están en [-1, 1] pero normalizados
        # Usar umbral más permisivo: 0.01% del máximo (antes 0.1%), mínimo 1e-9
        # Reducido de 0.001 a 0.0001 para capturar mejor la "cola" de la distribución
        threshold = max(psi_abs_sq_max * 0.0001, 1e-9)

        # Si todos los valores son muy pequeños (estado vacío real), usar umbral mínimo
        if psi_abs_sq_max < 1e-9:
            threshold = 1e-9  # Umbral mínimo para detectar cualquier actividad

        # Agregar partículas solo donde hay estado significativo
        particle_count = 0
        grid_size = dense_psi.shape[1]  # H == W

        # OPTIMIZACIÓN: Muestrear grid si es muy grande (>256x256)
        # Para grids grandes, no agregar partículas en cada célula (sería muy lento)
        sample_step = 1 if grid_size <= 256 else max(1, grid_size // 256)

        # Función interna para agregar partículas
        def add_particles_with_threshold(thresh):
            count = 0
            verified_count = 0
            for y in range(0, grid_size, sample_step):
                for x in range(0, grid_size, sample_step):
                    # Obtener estado en esta posición
                    cell_state = dense_psi[0, y, x, :]  # [d_state]
                    cell_density = psi_abs_sq[0, y, x, :].sum().item()

                    # Solo agregar partícula si tiene densidad significativa
                    if cell_density > thresh:
                        try:
                            # CRÍTICO: Verificar que cell_state no es cero antes de agregar
                            cell_state_abs_max = cell_state.abs().max().item()
                            if cell_state_abs_max < 1e-10:
                                continue

                            coord = atheria_core.Coord3D(x, y, 0)
                            # Asegurar que el tensor está en el dispositivo correcto
                            if cell_state.device != self.device:
                                cell_state = cell_state.to(self.device)

                            self.native_engine.add_particle(coord, cell_state)
                            count += 1

                            # VERIFICACIÓN INMEDIATA (muestreo): Verificar 1 de cada 100 para no impactar rendimiento
                            if count % 100 == 0 or count < 5:
                                check_state = self.native_engine.get_state_at(coord)
                                if check_state is not None:
                                    verified_count += 1
                        except Exception as e:
                            logging.warning(
                                f"⚠️ Error agregando partícula en ({x}, {y}): {e}"
                            )

            if count > 0 and verified_count == 0 and count < 5:
                logging.warning(
                    f"⚠️ Se intentaron agregar {count} partículas pero la verificación inmediata falló."
                )

            return count

        # Intentar agregar partículas
        particle_count = add_particles_with_threshold(threshold)

        # RETRY: Si no se agregaron partículas, intentar con umbral más bajo
        if particle_count == 0 and psi_abs_sq_max > 1e-10:
            logging.warning(
                f"⚠️ No se agregaron partículas con umbral {threshold:.6e}. Reintentando con umbral mínimo..."
            )
            threshold = 1e-10
            particle_count = add_particles_with_threshold(threshold)

        logging.info(
            f"✅ Estado inicial generado según INITIAL_STATE_MODE_INFERENCE: {particle_count} partículas activas agregadas al motor nativo (umbral={threshold:.6e})"
        )

        # CRÍTICO: Después de agregar partículas, verificar que realmente se agregaron
        # y se pueden recuperar antes de continuar
        try:
            # Verificar algunas partículas aleatorias que agregamos
            import random

            sample_size = min(10, particle_count)
            recovered_count = 0

            if particle_count > 0:
                # Tomar muestra de coordenadas donde agregamos partículas
                sample_coords = []
                for y in range(0, grid_size, sample_step):
                    for x in range(0, grid_size, sample_step):
                        if len(sample_coords) >= sample_size:
                            break
                        cell_state = dense_psi[0, y, x, :]
                        cell_density = psi_abs_sq[0, y, x, :].sum().item()
                        if cell_density > threshold:
                            sample_coords.append((x, y))
                    if len(sample_coords) >= sample_size:
                        break

                # Verificar que estas partículas se pueden recuperar
                for x, y in sample_coords[:sample_size]:
                    test_coord = atheria_core.Coord3D(x, y, 0)
                    test_state = self.native_engine.get_state_at(test_coord)
                    if test_state is not None:
                        test_abs = test_state.abs().max().item()
                        if test_abs > 1e-10:
                            recovered_count += 1
                            logging.info(
                                f"✅ Partícula recuperada en ({x}, {y}): max abs={test_abs:.6e}"
                            )
                        else:
                            logging.warning(
                                f"⚠️ Partícula en ({x}, {y}) tiene valores muy pequeños: max abs={test_abs:.6e} - probablemente vacío cuántico"
                            )
                    else:
                        logging.warning(
                            f"⚠️ No se pudo recuperar partícula en ({x}, {y}): get_state_at retornó None"
                        )

                # CRÍTICO: Si ninguna partícula es recuperable, hay un problema grave
                if recovered_count == 0 and particle_count > 0:
                    logging.error(
                        f"❌ CRÍTICO: Se agregaron {particle_count} partículas pero NINGUNA es recuperable."
                    )
                    logging.error(
                        f"❌ Esto indica que add_particle() en C++ NO está almacenando correctamente en matter_map_."
                    )

                    # FALLBACK ROBUSTO: Si falló la recuperación, intentar regenerar con método diferente
                    # En este caso, regenerate_initial_state volvería aquí (recursión infinita),
                    # así que simplemente reportamos el error y permitimos que el motor Python tome el control si es necesario
                    # o intentamos el método "deprecated" add_initial_particles que usa lógica más simple
                    logging.warning(
                        "⚠️ Intentando fallback a add_initial_particles (método simple)..."
                    )
                    try:
                        self.native_engine.clear()
                        # Generar partículas simples manualmente para asegurar que algo funciona
                        for _ in range(10):
                            fx = np.random.randint(0, grid_size)
                            fy = np.random.randint(0, grid_size)
                            fz = 0
                            f_state = torch.randn(
                                self.d_state, dtype=torch.complex64, device=self.device
                            )
                            f_coord = atheria_core.Coord3D(fx, fy, fz)
                            self.native_engine.add_particle(f_coord, f_state)

                        # Verificar fallback
                        f_count = self.native_engine.get_matter_count()
                        logging.info(
                            f"📊 Fallback status: {f_count} partículas agregadas"
                        )
                    except Exception as fb_err:
                        logging.error(f"❌ Fallback también falló: {fb_err}")

            # Verificar coordenadas activas del motor nativo
            if hasattr(self.native_engine, "get_active_coords"):
                try:
                    active_coords = self.native_engine.get_active_coords()
                    if active_coords and len(active_coords) > 0:
                        logging.info(
                            f"✅ Motor nativo tiene {len(active_coords)} coordenadas activas verificadas"
                        )
                    else:
                        logging.warning(
                            f"⚠️ Motor nativo no tiene coordenadas activas recuperables después de agregar {particle_count} partículas"
                        )
                except Exception as e:
                    logging.debug(f"⚠️ No se pudo obtener coordenadas activas: {e}")
        except Exception as verify_error:
            logging.warning(f"⚠️ Error verificando partículas agregadas: {verify_error}")

        # Marcar estado denso como stale para que se reconvierta cuando se necesite
        self._dense_state_stale = True

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
                logging.error(
                    f"❌ Error al cargar modelo TorchScript: {model_path} (error desconocido)"
                )
        return success

    def evolve_internal_state(self, step=None):
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
        roi_changed = (roi is not None and roi != self._last_conversion_roi) or (
            roi is None and self._last_conversion_roi is not None
        )

        if self._dense_state_stale or self.state.psi is None or roi_changed:
            self._update_dense_state_from_sparse(
                roi=roi, check_pause_callback=check_pause_callback
            )
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
        - Duplicate filtering: filtra coordenadas duplicadas del motor nativo

        Args:
            roi: Region of Interest (x_min, y_min, x_max, y_max) opcional.
                 Si se proporciona, solo convierte esa región.
            check_pause_callback: Función callback para verificar pausa.
                                 Debe retornar True si está pausado.
        """
        # Inicializar grid denso si no existe
        if self.state.psi is None:
            self.state.psi = torch.zeros(
                1,
                self.grid_size,
                self.grid_size,
                self.d_state,
                dtype=torch.complex64,
                device=self.device,
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
            logging.debug(
                f"Convirtiendo solo ROI: ({x_min}, {y_min}) - ({x_max}, {y_max}), {len(coords_list)} coordenadas"
            )
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
                if hasattr(self.native_engine, "get_active_coords"):
                    logging.debug(
                        "🔍 Solicitando coordenadas activas al motor nativo..."
                    )
                    active_coords_list = self.native_engine.get_active_coords()
                    logging.debug(
                        f"✅ Coordenadas activas recibidas: {len(active_coords_list) if active_coords_list else 0}"
                    )
                    if active_coords_list and len(active_coords_list) > 0:
                        # CRÍTICO: Filtrar duplicados Y coordenadas fuera del plano Z=0
                        # El motor nativo es 3D y puede propagar a Z!=0, pero la visualización es 2D
                        # Filtramos para quedarnos solo con el slice Z=0 y evitar "duplicados" en la proyección 2D
                        unique_coords_set = set()
                        unique_active_coords = []
                        z_filtered_count = 0

                        for coord in active_coords_list:
                            # Solo procesar slice Z=0
                            if coord.z != 0:
                                z_filtered_count += 1
                                continue

                            coord_tuple = (coord.x, coord.y, coord.z)
                            if coord_tuple not in unique_coords_set:
                                unique_coords_set.add(coord_tuple)
                                unique_active_coords.append(coord)

                        if z_filtered_count > 0:
                            logging.debug(
                                f"🔄 Filtrados {z_filtered_count} partículas fuera del plano Z=0"
                            )

                        if (
                            len(unique_active_coords)
                            < len(active_coords_list) - z_filtered_count
                        ):
                            logging.debug(
                                f"🔄 Filtrados {len(active_coords_list) - z_filtered_count - len(unique_active_coords)} duplicados reales en Z=0"
                            )

                        active_coords_list = unique_active_coords

                        # Filtrar coordenadas activas por ROI si aplica
                        if roi is not None:
                            x_min, y_min, x_max, y_max = roi
                            active_coords = [
                                coord
                                for coord in active_coords_list
                                if x_min <= coord.x < x_max and y_min <= coord.y < y_max
                            ]
                        else:
                            active_coords = active_coords_list
            except Exception as e:
                logging.warning(f"⚠️ Error obteniendo coordenadas activas: {e}")
                pass

            # Si tenemos coordenadas activas, usar solo esas (mucho más rápido)
            if active_coords is not None and len(active_coords) > 0:
                coords_to_process = active_coords
                BATCH_SIZE = 1000  # Batch más grande para coordenadas activas
                logging.debug(
                    f"Usando {len(active_coords)} coordenadas activas (ROI aplicado)"
                )
            else:
                # Sin coordenadas activas: usar lista completa (ROI o todo el grid)
                # OPTIMIZACIÓN: Sampling también para lista completa si es muy grande
                total_coords = len(coords_list)
                if total_coords > 100000:  # Para grids muy grandes
                    sample_rate = max(
                        1, total_coords // 50000
                    )  # Máximo 50k coordenadas
                    coords_to_process = coords_list[::sample_rate]
                    logging.info(
                        f"🔄 Sampling para lista completa: {len(coords_to_process)} de {total_coords} coordenadas (rate={sample_rate})"
                    )
                else:
                    coords_to_process = coords_list
                BATCH_SIZE = 1000  # Batch más grande para mejor rendimiento
                logging.debug(
                    f"Usando lista completa: {len(coords_to_process)} coordenadas"
                )

            # Procesar en batches con verificación de pausa
            non_zero_count = 0
            total_processed = 0
            max_abs_value = 0.0

            total_batches = (len(coords_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
            # Solo loguear inicio si son muchos batches
            if total_batches > 5:
                logging.info(
                    f"🔄 Iniciando conversión: {len(coords_to_process)} coordenadas en {total_batches} batches (tamaño={BATCH_SIZE})"
                )

            import time

            start_time = time.time()
            TIMEOUT = 30.0  # Timeout explícito para conversión (30s)

            for batch_idx, i in enumerate(range(0, len(coords_to_process), BATCH_SIZE)):
                # CRÍTICO: Verificar pausa cada batch para permitir pausa inmediata
                if check_pause_callback and check_pause_callback():
                    logging.debug("Conversión interrumpida por pausa")
                    return  # Salir temprano si está pausado

                # CRÍTICO: Verificar timeout global
                if time.time() - start_time > TIMEOUT:
                    logging.error(
                        f"❌ Timeout en conversión denso-disperso (> {TIMEOUT}s). Retornando estado parcial."
                    )
                    # Si falla, intentar al menos devolver lo que tenemos o un estado válido mínimo
                    break

                # Logging periódico para evitar bloqueos silenciosos (menos frecuente)
                if total_batches > 10 and (
                    batch_idx % 20 == 0 or batch_idx == total_batches - 1
                ):
                    logging.info(
                        f"📊 Conversión progreso: batch {batch_idx+1}/{total_batches} ({total_processed} procesadas)"
                    )

                batch_coords = coords_to_process[i : i + BATCH_SIZE]
                for coord_idx, coord in enumerate(batch_coords):
                    try:
                        state_tensor = self.native_engine.get_state_at(coord)

                        # Verificar que get_state_at retornó algo
                        if state_tensor is None:
                            continue

                        # Verificar shape antes de copiar
                        if state_tensor.shape == (self.d_state,):
                            # Verificar si el estado tiene datos válidos (no todo ceros)
                            abs_value = state_tensor.abs().max().item()
                            if abs_value > 1e-10:
                                non_zero_count += 1
                                max_abs_value = max(max_abs_value, abs_value)

                            # Mover a dispositivo correcto si es necesario
                            if state_tensor.device != self.device:
                                state_tensor = state_tensor.to(self.device)

                            # Copiar al estado denso (batch, H, W, d_state)
                            # CRÍTICO: Verificar que las coordenadas estén dentro del rango válido
                            if (
                                0 <= coord.x < self.grid_size
                                and 0 <= coord.y < self.grid_size
                            ):
                                self.state.psi[0, coord.y, coord.x] = state_tensor

                            total_processed += 1
                        else:
                            # Silencioso para no saturar logs
                            pass
                    except Exception as e:
                        # Silencioso para no saturar logs
                        pass

            # DEBUG: Logging de estadísticas de conversión (solo si hubo algo procesado)
            if total_processed > 0 and non_zero_count == 0:
                logging.warning(
                    f"⚠️ Conversión completada pero 0/{total_processed} coordenadas tienen estado no-cero. El motor nativo podría estar vacío."
                )

        except Exception as e:
            logging.error(
                f"❌ Error convirtiendo estado disperso a denso: {e}", exc_info=True
            )
            # En caso de error, mantener grid actual o inicializar vacío
            if self.state.psi is None:
                self.state.psi = torch.zeros(
                    1,
                    self.grid_size,
                    self.grid_size,
                    self.d_state,
                    dtype=torch.complex64,
                    device=self.device,
                )

    def get_visualization_data(self, viz_type: str = "density"):
        """
        Obtiene datos de visualización directamente del motor nativo.

        Args:
            viz_type: Tipo de visualización ("density", "phase", "energy")

        Returns:
            Tensor denso [H, W] con los valores calculados, o None si falla.
        """
        try:
            if hasattr(self.native_engine, "compute_visualization"):
                # Llamada directa a C++
                viz_tensor = self.native_engine.compute_visualization(viz_type)

                # Asegurar que esté en el mismo dispositivo que self.device
                if viz_tensor.device != self.device:
                    viz_tensor = viz_tensor.to(self.device)

                return viz_tensor
            else:
                logging.warning(
                    "Motor nativo no soporta compute_visualization. Usando fallback Python."
                )
                return None
        except Exception as e:
            logging.error(f"Error en compute_visualization nativo: {e}")
            return None

    def get_model_for_params(self):
        """Retorna el modelo para acceso a parámetros (compatibilidad)."""
        return self.original_model if self.original_model else self

    def compile_model(self):
        """Método de compatibilidad - el modelo nativo ya está compilado."""
        self.is_compiled = True
        logging.info("Modelo nativo: ya está optimizado (compilado en C++)")

    def regenerate_initial_state(self, cfg=None):
        """
        Regenera el estado inicial denso según INITIAL_STATE_MODE_INFERENCE y lo convierte
        al formato disperso del motor nativo. Esto respeta la ley M - las partículas emergen
        del estado inicial, no se agregan manualmente.

        Args:
            cfg: Configuración del experimento (opcional, usa self.cfg si no se proporciona)
        """
        from .. import config as global_cfg

        logging.info(
            "🔄 Regenerando estado inicial según INITIAL_STATE_MODE_INFERENCE..."
        )

        # Obtener modo de inicialización
        config = cfg if cfg is not None else getattr(self, "cfg", None)
        initial_mode = "complex_noise"  # Default
        if config is not None:
            initial_mode = getattr(
                config, "INITIAL_STATE_MODE_INFERENCE", "complex_noise"
            )
        else:
            initial_mode = getattr(
                global_cfg, "INITIAL_STATE_MODE_INFERENCE", "complex_noise"
            )

        # Limpiar motor nativo existente si es posible
        if hasattr(self, "native_engine") and self.native_engine is not None:
            try:
                # El motor nativo tiene método clear() según sparse_engine.h
                self.native_engine.clear()
                logging.debug("🧹 Motor nativo limpiado antes de regenerar estado")
            except Exception as e:
                logging.debug(
                    f"⚠️ Error limpiando motor nativo: {e}, continuando sin limpiar"
                )

        # Obtener configuración de grid scaling si está disponible
        base_state = None
        base_grid_size = None
        if config is not None:
            training_grid_size = getattr(config, "GRID_SIZE_TRAINING", None)
            if training_grid_size and training_grid_size < self.grid_size:
                # Crear estado base del tamaño de entrenamiento y replicarlo
                base_state_temp = QuantumState(
                    training_grid_size,
                    self.d_state,
                    self.device,
                    initial_mode=initial_mode,
                )
                base_state = base_state_temp.psi
                base_grid_size = training_grid_size
                logging.info(
                    f"🔄 Grid escalado: Creando estado base {training_grid_size}x{training_grid_size} para replicar en {self.grid_size}x{self.grid_size}"
                )

        # Regenerar estado cuántico denso según INITIAL_STATE_MODE_INFERENCE
        self.state = QuantumState(
            self.grid_size,
            self.d_state,
            self.device,
            initial_mode=initial_mode,
            base_state=base_state,
            base_grid_size=base_grid_size,
        )

        # CRÍTICO: Convertir estado denso inicial al formato disperso del motor nativo
        # Esto genera las partículas desde el estado denso (ley M respetada)
        self._initialize_native_state_from_dense(self.state.psi)

        # Marcar estado denso como stale para forzar reconversión si se necesita
        self._dense_state_stale = True

        logging.info(
            f"✅ Estado inicial regenerado según INITIAL_STATE_MODE_INFERENCE ({initial_mode})"
        )

    def add_initial_particles(self, num_particles: int = 10):
        """
        DEPRECADO: Este método es un hack temporal y NO respeta la ley M.

        Use regenerate_initial_state() en su lugar, que genera partículas desde el estado
        inicial denso según INITIAL_STATE_MODE_INFERENCE (las partículas emergen naturalmente).

        Este método solo se mantiene como fallback temporal para compatibilidad.

        Args:
            num_particles: Número de partículas a agregar (ignorado, se usa para logging)
        """
        import numpy as np

        logging.warning(
            f"⚠️ add_initial_particles() es DEPRECADO y NO respeta la ley M."
        )
        logging.warning(
            f"⚠️ Las partículas deberían emerger del estado inicial según INITIAL_STATE_MODE_INFERENCE."
        )
        logging.info(f"💡 Usando regenerate_initial_state() en su lugar...")

        # Usar el método correcto que respeta la ley M
        try:
            self.regenerate_initial_state()
        except Exception as e:
            logging.error(f"❌ Error regenerando estado inicial: {e}")
            logging.warning(f"⚠️ Fallback a add_initial_particles() (NO RECOMENDADO)")

            # Fallback solo si regenerate_initial_state falla
            logging.info(
                f"🛠️ Fallback: Agregando {num_particles} partículas aleatorias al motor nativo..."
            )

            # Generar partículas aleatorias en el grid
            for _ in range(
                min(num_particles, 100)
            ):  # Limitar a 100 para evitar bloqueos
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                z = 0  # Para 2D, z=0

                # Estado inicial con valores significativos
                initial_state = (
                    torch.randn(self.d_state, dtype=torch.complex64, device=self.device)
                    * 0.5
                )

                # Agregar al motor nativo
                coord = atheria_core.Coord3D(x, y, z)
                self.native_engine.add_particle(coord, initial_state)

            logging.info(
                f"✅ {min(num_particles, 100)} partículas aleatorias agregadas al motor nativo (fallback)"
            )

            # Marcar estado denso como stale
            self._dense_state_stale = True
            if self.state and self.state.psi is not None:
                self.state.psi = None

    def cleanup(self):
        """
        Limpia recursos del motor nativo de forma explícita.
        Debe llamarse antes de destruir el wrapper para evitar segfaults.
        """
        try:
            # Limpiar estado denso primero
            try:
                if hasattr(self, "state") and self.state is not None:
                    if hasattr(self.state, "psi") and self.state.psi is not None:
                        try:
                            self.state.psi = None
                        except Exception as psi_error:
                            logging.debug(f"Error limpiando state.psi: {psi_error}")
                    try:
                        self.state = None
                    except Exception as state_error:
                        logging.debug(f"Error limpiando state: {state_error}")
            except Exception as state_cleanup_error:
                logging.debug(f"Error durante limpieza de state: {state_cleanup_error}")

            # Limpiar referencias al motor nativo
            # El destructor de C++ se llamará automáticamente cuando no haya más referencias
            try:
                if hasattr(self, "native_engine") and self.native_engine is not None:
                    # Liberar referencias explícitamente
                    # CRÍTICO: Capturar cualquier error durante la liberación del motor C++
                    try:
                        self.native_engine = None
                    except Exception as native_cleanup_error:
                        logging.warning(
                            f"Error liberando native_engine: {native_cleanup_error}"
                        )
            except Exception as native_error:
                logging.warning(
                    f"Error durante limpieza de native_engine: {native_error}"
                )

            # Limpiar otras referencias de forma segura
            try:
                if hasattr(self, "dense_state_cache"):
                    self.dense_state_cache = None
            except Exception:
                pass  # Ignorar errores en limpieza de cache

            try:
                self.model_loaded = False
                self.step_count = 0
                self.last_delta_psi = None
                self.last_psi_input = None
                self.last_delta_psi_decay = None
            except Exception:
                pass  # Ignorar errores en limpieza de atributos simples

            logging.debug("NativeEngineWrapper limpiado correctamente")
        except Exception as e:
            logging.warning(
                f"Error durante cleanup de NativeEngineWrapper: {e}", exc_info=True
            )

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
