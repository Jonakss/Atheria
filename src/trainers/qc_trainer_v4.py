import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging

# Tensorboard es opcional
try:
from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("Tensorboard no está disponible. El logging de métricas se deshabilitará.")

# Importamos componentes del sistema
from ..engines.qca_engine import Aetheria_Motor
from ..physics.analysis.noise import QuantumNoiseInjector
from .. import config as global_cfg

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
                 lr=1e-4, grid_size=64, qca_steps=100, gamma_decay=0.01, model=None):
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
        """
        self.experiment_name = experiment_name
        self.device = device
        self.grid_size = grid_size
        self.qca_steps = qca_steps
        self.gamma_decay = gamma_decay
        
        # 1. Motor de Física (Ley M)
        # Si se proporciona un modelo ya instanciado, usarlo; sino, crear uno nuevo
        if model is not None:
            # Usar modelo ya instanciado (para checkpoints/transfer learning)
            d_state = model_params.get('d_state', 2) if isinstance(model_params, dict) else getattr(model_params, 'd_state', 2)
            self.motor = Aetheria_Motor(model, grid_size, d_state, device)
        elif model_class is not None and model_params is not None:
            # Crear nuevo modelo
            if isinstance(model_params, dict):
        self.motor = Aetheria_Motor(model_class(**model_params), grid_size, model_params['d_state'], device)
            else:
                # Si model_params es SimpleNamespace o similar
                params_dict = vars(model_params) if hasattr(model_params, '__dict__') else model_params
                self.motor = Aetheria_Motor(model_class(**params_dict), grid_size, params_dict['d_state'], device)
        else:
            raise ValueError("Debe proporcionarse 'model' o ('model_class' y 'model_params')")
        
        # 2. Inyector de Ruido (El "Enemigo" Evolutivo)
        self.noise_injector = QuantumNoiseInjector(device)
        
        # 3. Optimizador
        self.optimizer = optim.AdamW(self.motor.operator.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100, factor=0.5)
        
        # Directorios
        self.checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
        self.motor.operator.train()
        self.optimizer.zero_grad()
        
        # Estado inicial aleatorio (Sopa Primordial)
        self.motor.state._reset_state_random()
        psi = self.motor.state.psi
        psi_initial = psi.clone()
        
        psi_history = []
        
        # --- Curriculum de Ruido ---
        # Empezamos sin ruido (para aprender física básica)
        # Aumentamos el ruido linealmente hasta el episodio 2000
        max_noise = 0.05
        current_noise = min(max_noise, max_noise * (episode_num / 2000))
        
        # Simulación temporal
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
                
            psi_history.append(psi)
            
        # Calcular pérdida evolutiva
        loss, metrics = self.loss_function_evolutionary(psi_history, psi_initial)
        
        # Retropropagación
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.motor.operator.parameters(), 1.0) # Evitar explosión de gradientes
        self.optimizer.step()
        
        # Logging
        if episode_num % 10 == 0:
            if self.writer is not None:
            self.writer.add_scalar('Loss/Total', loss.item(), episode_num)
            self.writer.add_scalar('Metrics/Symmetry', metrics['symmetry'], episode_num)
            self.writer.add_scalar('Metrics/Survival', metrics['survival'], episode_num)
            self.writer.add_scalar('Environment/NoiseLevel', current_noise, episode_num)
            
        return loss.item(), metrics

    def save_checkpoint(self, episode, is_best=False):
        filename = "best_model.pth" if is_best else f"checkpoint_ep{episode}.pth"
        path = os.path.join(self.checkpoint_dir, filename)
        
        # Guardamos también el estado del optimizador para poder reanudar
        torch.save({
            'episode': episode,
            'model_state_dict': self.motor.operator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': 0.0 # Placeholder
        }, path)