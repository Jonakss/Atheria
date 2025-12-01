import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
import shutil
from typing import Dict, Optional
from types import SimpleNamespace

# Tensorboard es opcional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("Tensorboard no está disponible. El logging de métricas se deshabilitará.")

# Importamos componentes del sistema
from ..engines.qca_engine import Aetheria_Motor
from ..physics.noise import QuantumNoiseInjector
from .. import config as global_cfg
from ..utils.experiment_logger import ExperimentLogger
from src.cache import cache

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QC_Trainer_v4:
    """
    Entrenador para Atheria 4 (Cosmogénesis).
    Diferencias con V3:
    1. Inyecta ruido IonQ (Fase/Bit Flip) durante el entrenamiento.
    2. Función de pérdida multi-objetivo: Estabilidad + Simetría + Complejidad.
    3. Curriculum Learning: El ruido aumenta con los episodios.
    """
    def __init__(self, experiment_name, model_class=None, model_params=None, device=None, 
                 lr=1e-4, grid_size=64, qca_steps=100, gamma_decay=0.01, model=None,
                 max_checkpoints_to_keep=5, max_noise=0.05, engine_type=None):
        """
        Inicializa QC_Trainer_v4.
        
        Args:
            experiment_name: Nombre del experimento
            model_class: Clase del modelo (requerido si model no se proporciona)
            model_params: Parámetros del modelo como dict (requerido si model no se proporciona)
            device: Dispositivo (CPU/GPU)
            lr: Learning rate
            grid_size: Tamaño del grid
            qca_steps: Número de pasos QCA
            gamma_decay: Término Lindbladian (decaimiento)
            model: Modelo ya instanciado (opcional, para checkpoints/transfer learning)
            max_checkpoints_to_keep: Número máximo de mejores checkpoints a retener (por defecto 5)
            max_noise: Nivel máximo de ruido a inyectar (default 0.05)
            engine_type: Tipo de motor de física (opcional, por defecto usa global_cfg)
        """
        self.experiment_name = experiment_name
        self.device = device
        self.grid_size = grid_size
        self.qca_steps = qca_steps
        self.gamma_decay = gamma_decay
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.max_noise = max_noise
        
        # 1. Motor de Física (Ley M)
        # Si se proporciona un modelo ya instanciado, usarlo; sino, crear uno nuevo
        from src.motor_factory import get_motor
        
        # Preparar configuración para el factory
        # El factory espera un objeto config con GRID_SIZE_TRAINING, MODEL_PARAMS, etc.
        # Creamos un SimpleNamespace temporal si no tenemos uno completo
        factory_cfg = SimpleNamespace(
            GRID_SIZE_TRAINING=grid_size,
            MODEL_PARAMS=model_params if not isinstance(model_params, dict) else SimpleNamespace(**model_params),
            GAMMA_DECAY=gamma_decay,
            ENGINE_TYPE=engine_type if engine_type else global_cfg.ENGINE_TYPE # Usar inyectado o global
        )
        
        if model is not None:
            # Si ya es un motor, usarlo directamente
            if hasattr(model, 'evolve_internal_state'): # Check duck typing for Motor
                self.motor = model
            else:
                # Usar modelo ya instanciado (para checkpoints/transfer learning)
                # d_state se extrae dentro de get_motor o del config
                self.motor = get_motor(factory_cfg, device, model=model)
        elif model_class is not None and model_params is not None:
            # Crear nuevo modelo
            if isinstance(model_params, dict):
                model_instance = model_class(**model_params)
            else:
                # Si model_params es SimpleNamespace o similar
                params_dict = vars(model_params) if hasattr(model_params, '__dict__') else model_params
                model_instance = model_class(**params_dict)
                
            self.motor = get_motor(factory_cfg, device, model=model_instance)
        else:
            raise ValueError("Debe proporcionarse 'model' o ('model_class' y 'model_params')")
        
        # 2. Inyector de Ruido (El "Enemigo" Evolutivo)
        self.noise_injector = QuantumNoiseInjector(device)
        
        # 3. Optimizador
        # Obtener el modelo del motor (puede llamarse 'operator' o 'model' según el motor)
        self.model_operator = getattr(self.motor, 'operator', getattr(self.motor, 'model', None))
        
        if self.model_operator is None:
            # Si es LatticeEngine u otro sin modelo, advertir pero no fallar (a menos que se intente entrenar)
            logging.warning(f"⚠️ El motor {type(self.motor).__name__} no expone un modelo entrenable ('operator' o 'model').")
            self.optimizer = None
        else:
            self.optimizer = optim.AdamW(self.model_operator.parameters(), lr=lr, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100, factor=0.5)
        
        if self.optimizer is None:
             self.scheduler = None
        
        # 4. Mixed Precision Training (AMP) - reduce memoria ~50%
        self.use_amp = (str(device).startswith('cuda'))
        self.scaler = None
        if self.use_amp:
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                logging.info("🚀 Mixed Precision Training (FP16) habilitado - reduce uso de memoria ~50%")
            except ImportError:
                logging.warning("⚠️ torch.cuda.amp no disponible. Entrenando en FP32.")
                self.use_amp = False
        
        # Directorios
        self.checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Sistema Smart Save: Lista de checkpoints ordenados por calidad
        # Cada entrada: {'path': str, 'episode': int, 'metrics': dict, 'combined_metric': float}
        self.best_checkpoints: list = []
        
        # Experiment Logger para documentación automática
        self.doc_logger = ExperimentLogger(experiment_name)
        
        # Inicializar el logger con configuración del experimento
        config_dict = {
            'experiment_name': experiment_name,
            'model_architecture': getattr(model_params, 'MODEL_ARCHITECTURE', 'UNKNOWN') if hasattr(model_params, 'MODEL_ARCHITECTURE') else 'UNKNOWN',
            'lr': lr,
            'grid_size': grid_size,
            'qca_steps': qca_steps,
            'gamma_decay': gamma_decay,
            'd_state': model_params.get('d_state', 'UNKNOWN') if isinstance(model_params, dict) else getattr(model_params, 'd_state', 'UNKNOWN'),
            'max_checkpoints_to_keep': max_checkpoints_to_keep,
            'max_noise': max_noise
        }
        self.doc_logger.initialize_or_load(config_dict)
        
        # Tensorboard (opcional)
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(log_dir=os.path.join(global_cfg.LOGS_DIR, experiment_name))
            except Exception as e:
                logging.warning(f"No se pudo inicializar Tensorboard: {e}")

    def loss_function_evolutionary(self, psi_history, psi_initial):
        """
        Calcula qué tan "viva" está la simulación.
        Retorna: Loss total + Diccionario de métricas
        """
        psi_final = psi_history[-1]
        
        # A. Supervivencia (Energy Retention)
        # Queremos que la energía se mantenga a pesar del Gamma Decay y el Ruido.
        # No forzamos a que sea idéntica (permitimos metabolismo), pero penalizamos la muerte (0) o explosión (inf).
        final_energy = torch.sum(psi_final.abs().pow(2))
        initial_energy = torch.sum(psi_initial.abs().pow(2))
        target_energy = initial_energy * 0.8 # Permitimos un 20% de disipación natural
        loss_survival = torch.abs(final_energy - target_energy) / (initial_energy + 1e-6)
        
        # B. Simetría Local (IonQ Hypothesis)
        # Rotamos el estado 90 grados. Si es una partícula estable, debería parecerse a sí misma.
        # Esto fomenta la creación de "átomos" geométricos.
        psi_rot = torch.rot90(psi_final, 1, [1, 2]) # Rotar en ejes H, W
        loss_symmetry = torch.mean((psi_final.abs() - psi_rot.abs())**2)
        
        # C. Complejidad (Entropía Espacial)
        # Evitar el truco fácil de "llenar todo de ceros" o "llenar todo de unos".
        # Queremos islas de materia. Usamos la desviación estándar espacial.
        # Queremos MAXIMIZAR la complejidad -> MINIMIZAR el negativo.
        spatial_variance = torch.std(psi_final.abs().sum(dim=-1))
        loss_complexity = -torch.log(spatial_variance + 1e-6)

        # Ponderación de la Evolución (Ajustar según fase)
        total_loss = (10.0 * loss_survival) + (5.0 * loss_symmetry) + (1.0 * loss_complexity)
        
        metrics = {
            "survival": loss_survival.item(),
            "symmetry": loss_symmetry.item(),
            "complexity": loss_complexity.item()
        }
        return total_loss, metrics

    def train_episode(self, episode_num):
        if self.model_operator:
            self.model_operator.train()
        if self.optimizer:
            self.optimizer.zero_grad()
        
        # Estado inicial aleatorio (Sopa Primordial)
        if hasattr(self.motor, 'state') and hasattr(self.motor.state, '_reset_state_random'):
            self.motor.state._reset_state_random()
        elif hasattr(self.motor, '_initialize_links'):
            # For LatticeEngine
            self.motor.links = self.motor._initialize_links()
        if hasattr(self.motor, 'state') and hasattr(self.motor.state, 'psi'):
            psi = self.motor.state.psi
            psi_initial = psi.clone()
        elif hasattr(self.motor, 'get_initial_state'):
             # For engines like Lattice/Harmonic that generate state on fly or don't expose it directly
             psi_initial = self.motor.get_initial_state(self.batch_size)
             if hasattr(psi_initial, 'requires_grad_'):
                 psi_initial.requires_grad_(True)
        else:
             # Fallback or error
             raise AttributeError(f"Engine {type(self.motor).__name__} does not expose state.psi or get_initial_state")
        
        psi_history = []
        
        # --- Curriculum de Ruido ---
        # Empezamos sin ruido (para aprender física básica)
        # Aumentamos el ruido linealmente hasta el episodio 2000
        max_noise = self.max_noise
        current_noise = min(max_noise, max_noise * (episode_num / 2000))
        
        # Mixed Precision Training: autocast context para forward pass
        autocast_context = torch.cuda.amp.autocast() if self.use_amp else torch.cuda.amp.autocast(enabled=False)
        
        # Simulación temporal
        with autocast_context:
            for t in range(self.qca_steps):
                # 1. Ley M (Física)
                psi = self.motor.evolve_step(psi)
                
                # 2. Entorno Hostil (Ruido)
                if current_noise > 0.001:
                    # 80% Ruido de Fase (Z), 20% Ruido de Bit (X)
                    psi = self.noise_injector.apply_phase_flip(psi, rate=current_noise)
                    if np.random.random() < 0.2:
                        psi = self.noise_injector.apply_bit_flip(psi, rate=current_noise * 0.1)
                
                # Gradient Checkpointing para ahorrar VRAM en simulaciones largas
                # (Opcional, activo si pasos > 50)
                # if self.qca_steps > 50 and t % 10 == 0:
                #     psi = torch.utils.checkpoint.checkpoint(self.motor.operator, psi)
                
                # OPTIMIZACIÓN DE MEMORIA: Solo guardar estados necesarios para la pérdida
                # Solo guardamos: estado inicial, estados intermedios (cada N pasos), y estado final
                save_interval = max(1, self.qca_steps // 10)  # Guardar ~10 estados máximo
                if t == 0 or t == self.qca_steps - 1 or t % save_interval == 0:
                    psi_history.append(psi)
            
            # Asegurar que tenemos al menos el estado inicial y final
            if len(psi_history) == 0 or not torch.equal(psi_history[-1], psi):
                psi_history.append(psi)
            
            # Calcular pérdida evolutiva (dentro de autocast)
            loss, metrics = self.loss_function_evolutionary(psi_history, psi_initial)
        
        # OPTIMIZACIÓN DE MEMORIA: Limpiar psi_history después de calcular pérdida
        # Los tensores ya no se necesitan y pueden ocupar mucha memoria
        del psi_history
        import gc
        gc.collect()  # Forzar garbage collection para liberar memoria inmediatamente
        
        # Retropropagación con Mixed Precision (AMP) y manejo de memoria CUDA
        try:
            if self.use_amp and self.scaler:
                # Backward pass con gradient scaling (AMP)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.model_operator:
                    torch.nn.utils.clip_grad_norm_(self.model_operator.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Backward pass estándar (FP32)
                # Backward pass estándar (FP32)
                loss.backward()
                if self.model_operator:
                    torch.nn.utils.clip_grad_norm_(self.model_operator.parameters(), 1.0)
                self.optimizer.step()
        except torch.cuda.OutOfMemoryError as e:
            # Si hay error de memoria CUDA, limpiar caché y reintentar una vez
            logging.warning(f"⚠️ CUDA Out of Memory durante entrenamiento episodio {episode_num}. Limpiando caché...")
            torch.cuda.empty_cache()
            gc.collect()
            # Reintentar una vez después de limpiar
            try:
                if self.use_amp and self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.model_operator:
                        torch.nn.utils.clip_grad_norm_(self.model_operator.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.model_operator:
                        torch.nn.utils.clip_grad_norm_(self.model_operator.parameters(), 1.0)
                    self.optimizer.step()
                logging.info("✅ Recuperado después de limpiar caché CUDA")
            except torch.cuda.OutOfMemoryError:
                logging.error(f"❌ CUDA Out of Memory persistente en episodio {episode_num}. Deteniendo entrenamiento.")
                raise
        
        # Limpiar caché CUDA periódicamente (cada 10 episodios)
        if episode_num % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Logging
        if episode_num % 10 == 0:
            if self.writer is not None:
                self.writer.add_scalar('Loss/Total', loss.item(), episode_num)
                self.writer.add_scalar('Metrics/Symmetry', metrics['symmetry'], episode_num)
                self.writer.add_scalar('Metrics/Survival', metrics['survival'], episode_num)
                self.writer.add_scalar('Environment/NoiseLevel', current_noise, episode_num)
            
        return loss.item(), metrics

    def save_checkpoint(self, episode: int, current_metrics: Optional[Dict[str, float]] = None,
                       current_loss: Optional[float] = None, is_best: bool = False):
        """
        Guarda un checkpoint con sistema Smart Save.
        
        Implementa una política de retención que solo mantiene los N mejores checkpoints
        y el último, borrando automáticamente los antiguos para optimizar el uso de disco.
        
        Args:
            episode: Número de episodio
            current_metrics: Diccionario con métricas actuales (survival, symmetry, complexity)
            current_loss: Pérdida total actual
            is_best: Si este checkpoint es el mejor hasta ahora
        """
        # Valores por defecto si no se proporcionan
        if current_metrics is None:
            current_metrics = {}
        if current_loss is None:
            current_loss = 0.0
        
        # Calcular métrica combinada para ordenar checkpoints
        # Usamos la misma ponderación que en loss_function_evolutionary
        survival = current_metrics.get('survival', 0.0)
        symmetry = current_metrics.get('symmetry', 0.0)
        combined_metric = (10.0 * survival) + (5.0 * symmetry)  # Menor es mejor
        
        # Generar nombre de archivo
        if is_best:
            filename = "best_model.pth"
        else:
            filename = f"checkpoint_ep{episode}.pth"
        
        path = os.path.join(self.checkpoint_dir, filename)
        
        # Guardar checkpoint
        checkpoint_data = {
            'episode': episode,
            'model_state_dict': self.model_operator.state_dict() if self.model_operator else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': current_loss,
            'metrics': current_metrics.copy(),
            'combined_metric': combined_metric
        }
        torch.save(checkpoint_data, path)
        
        # Actualizar lista de mejores checkpoints
        checkpoint_entry = {
            'path': path,
            'episode': episode,
            'metrics': current_metrics.copy(),
            'loss': current_loss,
            'combined_metric': combined_metric
        }
        
        # Si es el mejor, añadirlo a la lista
        if is_best:
            self.best_checkpoints.append(checkpoint_entry)
            
            # Ordenar por métrica combinada (menor es mejor)
            self.best_checkpoints.sort(key=lambda x: x['combined_metric'])
            
            # Si excede el máximo, borrar los peores
            while len(self.best_checkpoints) > self.max_checkpoints_to_keep:
                worst_checkpoint = self.best_checkpoints.pop(-1)  # El último (peor)
                
                # Solo borrar si no es "best_model.pth" (mantener siempre el mejor actual)
                if os.path.basename(worst_checkpoint['path']) != "best_model.pth":
                    try:
                        if os.path.exists(worst_checkpoint['path']):
                            os.remove(worst_checkpoint['path'])
                            logging.info(f"🗑️  Checkpoint eliminado (Smart Save): {os.path.basename(worst_checkpoint['path'])} - Episodio {worst_checkpoint['episode']}")
                    except Exception as e:
                        logging.warning(f"Error al eliminar checkpoint antiguo: {e}")
        
        # OPTIMIZACIÓN DE DISCO: Limpiar checkpoints periódicos antiguos (no solo los mejores)
        # Mantener solo los últimos N checkpoints periódicos además de los mejores
        max_periodic_checkpoints = 10  # Mantener solo los últimos 10 checkpoints periódicos
        try:
            # Obtener todos los checkpoints periódicos (no best_model.pth ni last_checkpoint.pth)
            periodic_checkpoints = []
            if os.path.exists(self.checkpoint_dir):
                for filename in os.listdir(self.checkpoint_dir):
                    if filename.endswith('.pth') and filename.startswith('checkpoint_ep'):
                        filepath = os.path.join(self.checkpoint_dir, filename)
                        try:
                            # Obtener número de episodio del nombre
                            ep_num = int(filename.replace('checkpoint_ep', '').replace('.pth', ''))
                            periodic_checkpoints.append((ep_num, filepath))
                        except:
                            continue
            
            # Ordenar por episodio (más recientes primero)
            periodic_checkpoints.sort(reverse=True)
            
            # Eliminar los más antiguos si exceden el límite
            if len(periodic_checkpoints) > max_periodic_checkpoints:
                to_remove = periodic_checkpoints[max_periodic_checkpoints:]
                for ep_num, filepath in to_remove:
                    try:
                        # Verificar que no está en la lista de mejores
                        is_best = any(cp['path'] == filepath for cp in self.best_checkpoints)
                        if not is_best:
                            os.remove(filepath)
                            logging.info(f"🗑️  Checkpoint periódico antiguo eliminado: {os.path.basename(filepath)} - Episodio {ep_num}")
                    except Exception as e:
                        logging.warning(f"Error al eliminar checkpoint periódico antiguo: {e}")
        except Exception as e:
            logging.warning(f"Error limpiando checkpoints periódicos antiguos: {e}")
        
        # Registrar en el logger de documentación solo si es el mejor
        if is_best:
            self.doc_logger.log_result(
                episodes=episode,
                metrics=current_metrics,
                loss=current_loss,
                is_best=True,
                checkpoint_path=path
            )
            
            logging.info(
                f"✅ Checkpoint guardado (MEJOR): Episodio {episode} | "
                f"Loss: {current_loss:.6f} | "
                f"Survival: {survival:.6f} | Symmetry: {symmetry:.6f} | "
                f"Combined: {combined_metric:.6f}"
            )
        else:
            # Checkpoint periódico normal
            logging.debug(
                f"💾 Checkpoint guardado: Episodio {episode} | "
                f"Loss: {current_loss:.6f}"
            )
        
        # Siempre mantener el último checkpoint
        last_checkpoint_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pth")
        if path != last_checkpoint_path:
            try:
                shutil.copy2(path, last_checkpoint_path)
            except Exception as e:
                logging.warning(f"Error al copiar último checkpoint: {e}")

        # NUEVO: Cachear en Dragonfly para acceso rápido (Latest Checkpoint)
        if cache.enabled:
            try:
                # Usar una clave consistente para el último checkpoint de este experimento
                cache_key = f"checkpoint:{self.experiment_name}:latest"
                
                # Preparar datos para caché (asegurar que están en CPU)
                # No cacheamos todo el dict para ahorrar memoria/ancho de banda si no es necesario
                # Pero para resumir entrenamiento, necesitamos state_dicts
                
                # Convertir tensores a CPU para evitar problemas de pickle con CUDA
                cpu_checkpoint_data = {
                    'episode': episode,
                    'loss': current_loss,
                    'metrics': current_metrics,
                    'combined_metric': combined_metric,
                    # Convertir state_dicts a CPU recursivamente es costoso, 
                    # pero necesario si queremos independencia del device al cargar
                    # Por ahora confiamos en que torch.save/load manejan esto, 
                    # pero para pickle directo necesitamos cuidado.
                    # DragonflyCache usa pickle.
                }
                
                # Opción A: Solo cachear metadatos (rápido)
                # Opción B: Cachear modelo completo (lento pero permite fast-resume)
                
                # Vamos con Opción B pero optimizada: solo state_dict del modelo
                # (optimizador y scheduler pueden cargarse de disco si es necesario)
                
                # Mover state_dict a CPU
                model_state_cpu = {k: v.cpu() for k, v in self.motor.operator.state_dict().items()}
                cpu_checkpoint_data['model_state_dict'] = model_state_cpu
                
                # Guardar con TTL de 24 horas
                cache.set(cache_key, cpu_checkpoint_data, ttl=86400)
                logging.debug(f"💾 Checkpoint (modelo) cacheado en Dragonfly: {cache_key}")
                
            except Exception as e:
                logging.warning(f"⚠️ Error cacheando checkpoint en Dragonfly: {e}")