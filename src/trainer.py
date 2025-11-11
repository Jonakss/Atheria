# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
import gc
from torch.amp import GradScaler, autocast 

# ¡Importaciones relativas!
from .qca_engine import Aetheria_Motor
from .config import (
    DEVICE, EXPERIMENTS_DIR, EXPERIMENT_NAME, 
    LOAD_FROM_EXPERIMENT,
    PESO_QUIETUD, PESO_COMPLEJIDAD_LOCALIZADA,
    STAGNATION_WINDOW, MIN_LOSS_IMPROVEMENT, REACTIVATION_COUNT, 
    REACTIVATION_STATE_MODE, REACTIVATION_LR_MULTIPLIER, 
    GRADIENT_CLIP, STEPS_PER_EPISODE, PERSISTENCE_COUNT,
    D_STATE, CONTINUE_TRAINING
)

# ... (resto de la clase sin cambios hasta el __init__) ...
class QC_Trainer_v3:
    def __init__(self, motor: Aetheria_Motor, lr_rate: float, experiment_name: str):
        self.motor = motor
        if isinstance(self.motor.operator, nn.DataParallel):
            params_to_optimize = self.motor.operator.module.parameters()
        else:
            params_to_optimize = self.motor.operator.parameters()

        self.optimizer = optim.AdamW(
            params_to_optimize,
            lr=lr_rate,
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )
        
        self.scaler = GradScaler(device=DEVICE, enabled=(DEVICE.type == 'cuda'))
        
        self.experiment_name = experiment_name
        # ¡¡NUEVA RUTA!! Directorio de checkpoints específico para este experimento
        self.experiment_checkpoint_dir = os.path.join(EXPERIMENTS_DIR, self.experiment_name, "checkpoints")
        os.makedirs(self.experiment_checkpoint_dir, exist_ok=True)

        # ... (resto del __init__ sin cambios) ...

    def _save_checkpoint(self, episode, is_best=False):
        """Saves the training state to the specific experiment folder."""
        if is_best:
            filename = os.path.join(self.experiment_checkpoint_dir, f"qca_best_eps{episode}.pth")
        else:
            filename = os.path.join(self.experiment_checkpoint_dir, f"qca_checkpoint_eps{episode}.pth")

        if isinstance(self.motor.operator, nn.DataParallel):
            model_to_save = self.motor.operator.module
        else:
            model_to_save = self.motor.operator
        
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod
            
        model_state_dict = model_to_save.state_dict()

        state = {
            'episode': episode,
            'model_architecture': cfg.ACTIVE_QCA_OPERATOR,
            'd_state': cfg.D_STATE,
            'hidden_channels': cfg.HIDDEN_CHANNELS,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'stagnation_counter': self.stagnation_counter,
            'reactivation_counter': self.reactivation_counter,
            'history': self.history,
            'scaler_state_dict': self.scaler.state_dict()
        }
        torch.save(state, filename)
        print(f"\n[Checkpoint saved to: {filename}]")

    def _load_checkpoint(self):
        """
        Carga un checkpoint desde la nueva estructura de directorios.
        - Si `CONTINUE_TRAINING` es True, reanuda el *mismo* experimento.
        - Si `LOAD_FROM_EXPERIMENT` está definido, carga *solo los pesos* de ese experimento
          para hacer Transfer Learning.
        """
        search_path = None
        load_mode = "scratch" # ('scratch', 'resume', 'transfer')

        if CONTINUE_TRAINING:
            # Modo 1: Reanudar este experimento desde su carpeta de checkpoints
            search_path = os.path.join(self.experiment_checkpoint_dir, "qca_checkpoint_eps*.pth")
            load_mode = "resume"
        elif LOAD_FROM_EXPERIMENT:
            # Modo 2: Cargar pesos de la carpeta de checkpoints de un experimento anterior
            source_exp_dir = os.path.join(EXPERIMENTS_DIR, LOAD_FROM_EXPERIMENT, "checkpoints")
            search_path = os.path.join(source_exp_dir, "*_FINAL.pth") # Asumiendo que guardas un _FINAL
            load_mode = "transfer"

        if not search_path:
            print(f"Iniciando desde cero (CONTINUE_TRAINING=False, LOAD_FROM_EXPERIMENT=None).")
            return

        try:
            list_of_files = glob.glob(search_path)
            if not list_of_files:
                # Fallback: si no hay _FINAL.pth, busca el 'best'
                if load_mode == "transfer":
                    source_exp_dir = os.path.join(EXPERIMENTS_DIR, LOAD_FROM_EXPERIMENT, "checkpoints")
                    search_path = os.path.join(source_exp_dir, "qca_best_eps*.pth")
                    list_of_files = glob.glob(search_path)

                if not list_of_files:
                    print(f"No se encontraron checkpoints en '{os.path.dirname(search_path)}'. Empezando desde cero.")
                    return

            latest_file = max(list_of_files, key=os.path.getmtime)
            print(f"Cargando checkpoint: {latest_file} (Modo: {load_mode})")
            
            # Para transfer, solo necesitamos los pesos, no el estado del optimizador, etc.
            weights_only = load_mode == "transfer"
            checkpoint = torch.load(latest_file, map_location=DEVICE, weights_only=weights_only)

            target_model = self.motor.operator
            if isinstance(target_model, nn.DataParallel): target_model = target_model.module
            if hasattr(target_model, '_orig_mod'): target_model = target_model._orig_mod
            
            state_dict = checkpoint if weights_only else checkpoint['model_state_dict']

            is_dataparallel_saved = next(iter(state_dict)).startswith('module.')
            if is_dataparallel_saved:
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                target_model.load_state_dict(new_state_dict, strict=False)
            else:
                target_model.load_state_dict(state_dict, strict=False)

            print("Pesos del modelo cargados exitosamente (strict=False).")

            if load_mode == "resume":
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.current_episode = checkpoint['episode'] + 1
                self.best_loss = checkpoint['best_loss']
                self.history = checkpoint.get('history', self.history)
                self.stagnation_counter = checkpoint.get('stagnation_counter', 0)
                self.reactivation_counter = checkpoint.get('reactivation_counter', 0)
                print(f"Checkpoint reanudado. Empezando desde episodio {self.current_episode}.")
            else:
                print("Transferencia de pesos completada. Empezando nuevo entrenamiento desde episodio 0.")

        except Exception as e:
            print(f"Error al cargar checkpoint: {e}. Empezando desde cero.")
            self.current_episode = 0
            self.history = {k: [] for k in self.history.keys()}
            self.best_loss = float('inf')
            self.stagnation_counter = 0
            self.reactivation_counter = 0
            self.scaler = GradScaler(device=DEVICE, enabled=(DEVICE.type == 'cuda'))

    def check_stagnation_and_reactivate(self, total_episodes):
        """Checks for training stagnation and triggers reactivation if configured."""
        current_loss = self.history['Loss'][-1] if self.history['Loss'] else float('inf')
        if not np.isnan(current_loss) and not np.isinf(current_loss) and current_loss < (self.best_loss - MIN_LOSS_IMPROVEMENT):
            self.best_loss = current_loss
            self.stagnation_counter = 0
            return False 
        else:
            self.stagnation_counter += 1
            # ... (Lógica de estancamiento y reactivación, sin cambios) ...
            if self.stagnation_counter >= STAGNATION_WINDOW:
                if self.reactivation_counter < REACTIVATION_COUNT:
                    self.reactivation_counter += 1
                    # ... (lógica de reseteo de estado y LR) ...
                    print(f"Attempting reactivation {self.reactivation_counter}/{REACTIVATION_COUNT}...")
                    self.motor.state._reset_state_random() # <--- Simplificado
                    print("-> Resetting state with random noise.")
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = current_lr * REACTIVATION_LR_MULTIPLIER
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"-> Learning rate adjusted from {current_lr:.2e} to {new_lr:.2e}.")
                    self.stagnation_counter = 0
                    return False
                else:
                    print(f"Maximum number of reactivations ({REACTIVATION_COUNT}) reached.")
                    return True
            return False

    def train_episode(self, total_episodes):
        """Runs one full training episode (BPTT-k) with mixed precision."""
        self.motor.state._reset_state_random()

        episode_total_loss = 0.0
        bptt_cumulative_loss = 0.0
        valid_steps = 0
        
        # --- ¡¡MODIFICADO!! Un solo tensor 'psi' ---
        current_psi = self.motor.state.psi.clone().requires_grad_(True).to(DEVICE)
        
        last_R_quietud, last_R_complex = (float('nan'),)*2
        self.optimizer.zero_grad()

        for t in range(STEPS_PER_EPISODE):
            with autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == 'cuda')):
                if torch.isnan(current_psi).any() or torch.isinf(current_psi).any():
                    print(f"⚠️  NaN/Inf detected in state at step {t} of episode {self.current_episode}.")
                    episode_total_loss = float('nan')
                    break

                prev_psi_detached = current_psi.detach()
                
                # [B, H, W, C] -> [B, C, H, W]
                x_cat = current_psi.permute(0, 3, 1, 2)

                # --- 1. Aplicar Ley M (Unitaria) ---
                # delta_psi tiene forma [B, C, H, W]
                delta_psi = self.motor.operator(x_cat)
                
                if torch.isnan(delta_psi).any() or torch.isinf(delta_psi).any():
                    print(f"⚠️  NaN/Inf detected in delta_psi at step {t} of episode {self.current_episode}.")
                    episode_total_loss = float('nan')
                    break
                
                # --- 2. Aplicar Método de Euler ---
                # [B, H, W, C] + [B, C, H, W] -> [B, H, W, C]
                next_psi = current_psi + delta_psi.permute(0, 2, 3, 1)

                # ¡¡NORMALIZACIÓN ELIMINADA!!
                # La física de la U-Net (A - A.T) se encarga de esto.

                # --- 3. ¡¡NUEVA LÓGICA DE RECOMPENSA!! ---
                
                # Calcular la norma (energía/densidad)
                # Suma los cuadrados de los 42 canales en cada celda
                density_map = torch.sum(next_psi.pow(2), dim=-1).squeeze(0) # [H, W]
                
                # Calcular el cambio (actividad)
                change_vector = next_psi - prev_psi_detached
                change_magnitude_per_cell = torch.sum(change_vector.pow(2), dim=-1).squeeze(0) # [H, W]

                # R_Quietud: Fomenta el "vacío".
                R_Quietud = -change_magnitude_per_cell.mean() 
                
                # R_Complejidad: Fomenta la "materia".
                R_Complejidad_Localizada = density_map.std()
                
                last_R_quietud = R_Quietud.item()
                last_R_complex = R_Complejidad_Localizada.item()

                recompensa_total = (PESO_QUIETUD * R_Quietud) + \
                                   (PESO_COMPLEJIDAD_LOCALIZADA * R_Complejidad_Localizada)
                                   
                step_loss = -recompensa_total
                # --- FIN DE LA NUEVA LÓGICA ---
            
            if torch.isnan(step_loss) or torch.isinf(step_loss):
                print(f"⚠️  NaN/Inf detected in step_loss at step {t} of episode {self.current_episode}.")
                episode_total_loss = float('nan')
                break

            bptt_cumulative_loss = bptt_cumulative_loss + (step_loss / PERSISTENCE_COUNT)
            if not torch.isnan(step_loss):
                episode_total_loss += step_loss.item()
            valid_steps += 1

            if (t + 1) % PERSISTENCE_COUNT == 0 or (t + 1) == STEPS_PER_EPISODE:
                if bptt_cumulative_loss != 0 and not torch.isnan(bptt_cumulative_loss) and not torch.isinf(bptt_cumulative_loss):
                    
                    self.scaler.scale(bptt_cumulative_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    all_grads_valid = True
                    params_to_check = [p for p in self.motor.operator.parameters() if p.requires_grad]
                    for p in params_to_check:
                        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                            all_grads_valid = False
                            break
                    
                    if not all_grads_valid:
                        print(f"⚠️  NaN/Inf gradient detected at step {t} of episode {self.current_episode}. Skipping optimizer step.")
                        self.optimizer.zero_grad(set_to_none=True)
                        self.gradient_norms.append(float('inf'))
                        episode_total_loss = float('nan')
                        self.scaler.update() # Resetear el scaler
                        break 
                    
                    params_to_clip = [p for p in params_to_check if p.grad is not None] 
                    if params_to_clip:
                        total_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, GRADIENT_CLIP)
                        self.gradient_norms.append(total_norm.item())
                    else:
                        self.gradient_norms.append(0.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True) 
                
                else: 
                    if not (np.isnan(episode_total_loss) or np.isinf(episode_total_loss)):
                        print(f"⚠️  BPTT cumulative loss is NaN/Inf/Zero at step {t}. Skipping backward pass.")
                    self.gradient_norms.append(0.0)
                    self.optimizer.zero_grad() 

                bptt_cumulative_loss = 0.0
                current_psi = next_psi.detach().to(DEVICE).requires_grad_(True)
            
            else: 
                current_psi = next_psi

        # --- Fin del Episodio ---
        avg_loss = episode_total_loss / max(valid_steps, 1)

        self.history['Loss'].append(avg_loss)
        self.history['R_Quietud'].append(last_R_quietud)
        self.history['R_Complejidad_Localizada'].append(last_R_complex)
        self.history['Gradient_Norm'].append(np.mean(self.gradient_norms) if self.gradient_norms else 0.0)
        self.gradient_norms = []
        
        self.current_episode += 1
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return avg_loss