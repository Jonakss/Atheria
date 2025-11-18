# src/qc_trainer.py
import torch
import torch.optim as optim
import logging
import os
import time
from types import SimpleNamespace
from datetime import datetime

from . import config as global_cfg
from .utils import get_latest_checkpoint, save_experiment_config

class QC_Trainer_v3:
    """
    Entrenador para modelos QCA que gestiona el ciclo de entrenamiento completo.
    """
    def __init__(self, motor, lr_rate, global_cfg, exp_cfg):
        """
        Inicializa el entrenador.
        
        Args:
            motor: Instancia de Aetheria_Motor con el modelo
            lr_rate: Learning rate
            global_cfg: Configuración global
            exp_cfg: Configuración del experimento (SimpleNamespace)
        """
        self.motor = motor
        self.lr_rate = lr_rate
        self.global_cfg = global_cfg
        self.exp_cfg = exp_cfg
        
        # Configurar optimizador
        self.optimizer = optim.Adam(
            self.motor.operator.parameters(),
            lr=self.lr_rate
        )
        
        # Parámetros de entrenamiento desde exp_cfg
        self.total_episodes = exp_cfg.TOTAL_EPISODES
        self.qca_steps = exp_cfg.QCA_STEPS_TRAINING
        self.batch_size = getattr(exp_cfg, 'BATCH_SIZE_TRAINING', 4)
        self.save_every = getattr(exp_cfg, 'SAVE_EVERY_EPISODES', 50)
        self.gradient_clip = getattr(exp_cfg, 'GRADIENT_CLIP', 1.0)
        
        # Parámetros de recompensa
        self.peso_quietud = getattr(exp_cfg, 'PESO_QUIETUD', 10.0)
        self.peso_complejidad = getattr(exp_cfg, 'PESO_COMPLEJIDAD_LOCALIZADA', 1.0)
        
        # Directorio de checkpoints
        self.checkpoint_dir = os.path.join(
            global_cfg.TRAINING_CHECKPOINTS_DIR,
            exp_cfg.EXPERIMENT_NAME
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Episodio inicial (para continuar entrenamiento)
        # Si se proporcionó START_EPISODE en exp_cfg, usarlo (ya viene del checkpoint)
        self.start_episode = getattr(exp_cfg, 'START_EPISODE', 0)
        
        # Cargar checkpoint si existe y es necesario (transfer learning)
        if hasattr(exp_cfg, 'LOAD_FROM_EXPERIMENT') and exp_cfg.LOAD_FROM_EXPERIMENT:
            self._load_from_experiment(exp_cfg.LOAD_FROM_EXPERIMENT)
            # Para transfer learning, siempre empezamos desde 0
            self.start_episode = 0
        
        # Si estamos continuando entrenamiento y ya cargamos el checkpoint en pipeline_train,
        # el modelo ya está cargado y solo necesitamos cargar el estado del optimizer
        if hasattr(exp_cfg, 'CONTINUE_TRAINING') and exp_cfg.CONTINUE_TRAINING and self.start_episode > 0:
            checkpoint_path = get_latest_checkpoint(exp_cfg.EXPERIMENT_NAME)
            if checkpoint_path:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.motor.device)
                    # El modelo ya está cargado en pipeline_train, solo cargar optimizer
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logging.info(f"Estado del optimizador cargado desde checkpoint")
                    # El start_episode ya viene de exp_cfg
                    logging.info(f"Continuando entrenamiento desde episodio {self.start_episode}")
                except Exception as e:
                    logging.warning(f"Error al cargar estado del optimizador: {e}")
        
        logging.info(f"QC_Trainer_v3 inicializado. Total episodios: {self.total_episodes}, empezando en: {self.start_episode}")
    
    def _load_from_experiment(self, exp_name):
        """Carga pesos desde un experimento anterior (transfer learning)."""
        checkpoint_path = get_latest_checkpoint(exp_name)
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.motor.device)
                # Cargar pesos con strict=False para permitir cambios de arquitectura
                self.motor.operator.load_state_dict(
                    checkpoint['model_state_dict'],
                    strict=False
                )
                # Para transfer learning, siempre empezamos desde 0
                logging.info(f"Pesos cargados desde '{exp_name}' (transfer learning). Iniciando desde episodio 0.")
            except Exception as e:
                logging.warning(f"Error al cargar pesos desde '{exp_name}': {e}. Iniciando desde cero.")
    
    def compute_reward(self, psi_initial, psi_final):
        """
        Calcula la función de recompensa basada en quietud y complejidad localizada.
        """
        # Quietud: penalizar cambios grandes
        delta_psi = psi_final - psi_initial
        delta_norm = torch.mean(torch.abs(delta_psi))
        reward_quietud = -self.peso_quietud * delta_norm
        
        # Complejidad localizada: promover estructura local sin oscilaciones globales
        # Medir varianza espacial (estructura local)
        spatial_var = torch.var(psi_final.abs().flatten(start_dim=1), dim=1)
        # Penalizar varianza muy alta (ruido) o muy baja (uniformidad)
        target_var = 0.1  # Valor objetivo de varianza
        reward_complejidad = -self.peso_complejidad * torch.abs(spatial_var - target_var).mean()
        
        total_reward = reward_quietud + reward_complejidad
        return total_reward, reward_quietud, reward_complejidad
    
    def train_episode(self, episode):
        """
        Entrena un episodio completo.
        
        Returns:
            (loss, reward, reward_quietud, reward_complejidad)
        """
        self.motor.operator.train()
        self.optimizer.zero_grad()
        
        # Inicializar estados del batch
        psi_initial = self.motor.get_initial_state(self.batch_size)
        psi_initial.requires_grad_(True)
        
        # Evolucionar el estado
        psi_final = psi_initial
        for step in range(self.qca_steps):
            psi_final = self.motor.evolve_step(psi_final)
            # Limpiar caché periódicamente durante la evolución (cada 5 pasos)
            if torch.cuda.is_available() and step % 5 == 0 and step > 0:
                torch.cuda.empty_cache()
        
        # IMPORTANTE: Resetear estados de memoria después de cada episodio de entrenamiento
        # para evitar problemas con backward a través de múltiples episodios
        if hasattr(self.motor.state, 'h_state') and self.motor.state.h_state is not None:
            self.motor.state.h_state = None
        if hasattr(self.motor.state, 'c_state') and self.motor.state.c_state is not None:
            self.motor.state.c_state = None
        
        # Calcular recompensa
        total_reward, reward_quietud, reward_complejidad = self.compute_reward(
            psi_initial, psi_final
        )
        
        # El objetivo es maximizar la recompensa, así que minimizamos -reward
        loss = -total_reward
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.motor.operator.parameters(),
                self.gradient_clip
            )
        
        self.optimizer.step()
        
        # Guardar valores antes de liberar memoria
        loss_val = loss.item()
        reward_val = total_reward.item()
        reward_quietud_val = reward_quietud.item()
        reward_complejidad_val = reward_complejidad.item()
        
        # Limpiar memoria GPU después de cada episodio
        if torch.cuda.is_available():
            # Liberar tensores intermedios
            del psi_initial, psi_final, loss, total_reward, reward_quietud, reward_complejidad
            # Limpiar caché de GPU
            torch.cuda.empty_cache()
            # Sincronizar para asegurar que la limpieza se complete
            torch.cuda.synchronize()
        
        return loss_val, reward_val, reward_quietud_val, reward_complejidad_val
    
    def save_checkpoint(self, episode, loss, reward):
        """Guarda un checkpoint del entrenamiento."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_ep{episode}.pth"
        )
        
        # Convertir exp_cfg a dict recursivamente para evitar problemas de serialización
        def sns_to_dict(obj):
            if isinstance(obj, SimpleNamespace):
                return {key: sns_to_dict(value) for key, value in vars(obj).items()}
            elif isinstance(obj, dict):
                return {key: sns_to_dict(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sns_to_dict(item) for item in obj]
            else:
                # Para tipos primitivos y objetos no serializables, convertir a str si es necesario
                try:
                    import json
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Obtener el state_dict del modelo, manejando modelos compilados
        model_state_dict = self.motor.operator.state_dict()
        
        # Si el modelo está compilado, las claves tienen prefijo "_orig_mod."
        # Remover el prefijo para guardar el modelo sin compilar
        if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
            logging.info("Modelo compilado detectado, removiendo prefijo '_orig_mod.' al guardar checkpoint")
            cleaned_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key.replace('_orig_mod.', '', 1)
                    cleaned_state_dict[new_key] = value
                else:
                    cleaned_state_dict[key] = value
            model_state_dict = cleaned_state_dict
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'reward': reward,
            'config': sns_to_dict(self.exp_cfg)
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint guardado: {checkpoint_path}")
    
    def run_training_loop(self):
        """Ejecuta el bucle principal de entrenamiento."""
        logging.info("Iniciando bucle de entrenamiento...")
        
        # Tracking de tiempo
        training_start_time = time.time()
        last_loss = None
        last_reward = None
        last_episode = None
        
        try:
            for episode in range(self.start_episode, self.total_episodes):
                try:
                    loss, reward, reward_quietud, reward_complejidad = self.train_episode(episode)
                    last_loss = loss
                    last_reward = reward
                    last_episode = episode
                    
                    # Limpiar memoria GPU cada 10 episodios (limpieza adicional)
                    if torch.cuda.is_available() and episode % 10 == 0:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Log cada 10 episodios
                    if episode % 10 == 0:
                        logging.info(
                            f"Episodio {episode}/{self.total_episodes} | "
                            f"Loss: {loss:.6f} | Reward: {reward:.6f} | "
                            f"Quietud: {reward_quietud:.6f} | Complejidad: {reward_complejidad:.6f}"
                        )
                        print(
                            f"Episodio {episode}/{self.total_episodes} | "
                            f"Loss: {loss:.6f} | Reward: {reward:.6f} | "
                            f"Quietud: {reward_quietud:.6f} | Complejidad: {reward_complejidad:.6f}",
                            flush=True
                        )
                    
                    # Guardar checkpoint periódicamente
                    if (episode + 1) % self.save_every == 0:
                        self.save_checkpoint(episode, loss, reward)
                        
                except KeyboardInterrupt:
                    # Si se interrumpe con Ctrl+C, guardar checkpoint antes de salir
                    logging.info("Entrenamiento interrumpido por el usuario. Guardando checkpoint...")
                    if last_episode is not None:
                        try:
                            self.save_checkpoint(last_episode, last_loss, last_reward)
                            logging.info(f"Checkpoint guardado en episodio {last_episode} antes de cerrar.")
                        except Exception as e:
                            logging.error(f"Error al guardar checkpoint de emergencia: {e}")
                    raise  # Re-lanzar para que el proceso termine
                except Exception as e:
                    logging.error(f"Error en episodio {episode}: {e}", exc_info=True)
                    # Continuar con el siguiente episodio
                    continue
            
            # Calcular tiempo total de entrenamiento
            training_end_time = time.time()
            training_duration = training_end_time - training_start_time
            
            logging.info("Entrenamiento completado.")
            print("✅ Entrenamiento completado.", flush=True)
            
            # Guardar checkpoint final si no se guardó en la última iteración
            if last_episode is not None and (last_episode + 1) % self.save_every != 0:
                try:
                    self.save_checkpoint(last_episode, last_loss, last_reward)
                except Exception as e:
                    logging.warning(f"No se pudo guardar checkpoint final: {e}")
            
            # Actualizar config con tiempo de entrenamiento
            try:
                from .utils import load_experiment_config
                exp_config = load_experiment_config(self.exp_cfg.EXPERIMENT_NAME)
                if exp_config:
                    config_dict = {}
                    for key, value in vars(exp_config).items():
                        if isinstance(value, SimpleNamespace):
                            config_dict[key] = {k: v for k, v in vars(value).items()}
                        else:
                            config_dict[key] = value
                    
                    # Actualizar tiempos
                    existing_total = config_dict.get('total_training_time', 0)
                    config_dict['total_training_time'] = existing_total + training_duration
                    config_dict['last_training_time'] = datetime.now().isoformat()
                    config_dict['updated_at'] = datetime.now().isoformat()
                    
                    save_experiment_config(self.exp_cfg.EXPERIMENT_NAME, config_dict, is_update=True)
                    logging.info(f"Tiempo total de entrenamiento: {training_duration:.2f} segundos ({training_duration/3600:.2f} horas)")
            except Exception as e:
                logging.warning(f"No se pudo actualizar tiempo de entrenamiento en config: {e}")
        except KeyboardInterrupt:
            # Manejar Ctrl+C a nivel del bucle principal
            training_end_time = time.time()
            training_duration = training_end_time - training_start_time
            
            logging.info("Entrenamiento interrumpido. Guardando checkpoint final...")
            if last_episode is not None:
                try:
                    self.save_checkpoint(last_episode, last_loss, last_reward)
                    logging.info(f"Checkpoint de emergencia guardado en episodio {last_episode}.")
                except Exception as e:
                    logging.error(f"Error al guardar checkpoint de emergencia: {e}")
            
            # Actualizar tiempo parcial de entrenamiento
            try:
                from .utils import load_experiment_config
                exp_config = load_experiment_config(self.exp_cfg.EXPERIMENT_NAME)
                if exp_config:
                    config_dict = {}
                    for key, value in vars(exp_config).items():
                        if isinstance(value, SimpleNamespace):
                            config_dict[key] = {k: v for k, v in vars(value).items()}
                        else:
                            config_dict[key] = value
                    
                    existing_total = config_dict.get('total_training_time', 0)
                    config_dict['total_training_time'] = existing_total + training_duration
                    config_dict['last_training_time'] = datetime.now().isoformat()
                    config_dict['updated_at'] = datetime.now().isoformat()
                    
                    save_experiment_config(self.exp_cfg.EXPERIMENT_NAME, config_dict, is_update=True)
            except Exception as e:
                logging.warning(f"No se pudo actualizar tiempo de entrenamiento: {e}")
            
            raise

