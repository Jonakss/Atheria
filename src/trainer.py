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
    DEVICE, CHECKPOINT_DIR, EXPERIMENT_NAME, # <-- ¡NUEVO! Importa el nombre
    ALPHA_START, ALPHA_END, GAMMA_START, GAMMA_END,
    BETA_CAUSALITY, LAMBDA_ACTIVITY_VAR, LAMBDA_VELOCIDAD, TARGET_STD_DENSITY,
    EXPLOSION_THRESHOLD, EXPLOSION_PENALTY_MULTIPLIER, STAGNATION_WINDOW,
    MIN_LOSS_IMPROVEMENT, REACTIVATION_COUNT, REACTIVATION_STATE_MODE,
    REACTIVATION_LR_MULTIPLIER, GRADIENT_CLIP, STEPS_PER_EPISODE,
    PERSISTENCE_COUNT, CHECKPOINTS_DIR_RELATIVE
)

# ------------------------------------------------------------------------------
# 2.1: QC_Trainer_v3 Class
# ------------------------------------------------------------------------------
class QC_Trainer_v3:
    def __init__(self, motor: Aetheria_Motor, lr_rate: float, experiment_name: str):
        self.motor = motor
        self.optimizer = optim.AdamW(
            self.motor.operator.parameters(), # (Simplificado, DataParallel maneja esto)
            lr=lr_rate,
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )
        self.scaler = GradScaler(device=DEVICE, enabled=(DEVICE.type == 'cuda'))
        
        # --- ¡¡NUEVO!! Guardar el nombre del experimento ---
        self.experiment_name = experiment_name
        # --- ¡¡MODIFICADO!! Usar directorio de checkpoints fuera del experimento ---
        self.experiment_checkpoint_dir = os.path.abspath(os.path.join(CHECKPOINT_DIR, CHECKPOINTS_DIR_RELATIVE.replace('experiment_name', self.experiment_name)))
        os.makedirs(self.experiment_checkpoint_dir, exist_ok=True)
        print(f"Checkpoints se guardarán en: {self.experiment_checkpoint_dir}")
        # --------------------------------------------------

        self.history = {
            'Loss': [], 'R_Density_Target': [], 'R_Causalidad': [],
            'R_Stability': [], 'P_Explosion': [], 'Gradient_Norm': [],
            'R_Activity_Var': [], 'R_Velocidad': []
        }
        self.current_episode = 0
        self.best_loss = float('inf')
        self.stagnation_counter = 0
        self.reactivation_counter = 0
        self.gradient_norms = []

    # ... ( _calculate_annealed_alpha_gamma no cambia) ...
    def _calculate_annealed_alpha_gamma(self, total_episodes):
        total_episodes_for_annealing = total_episodes * 0.75
        progress = min(1.0, self.current_episode / max(1.0, total_episodes_for_annealing))
        alpha_progress = 1 - (1 - progress) ** 1.5
        gamma_progress = progress ** 0.7
        current_alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * alpha_progress
        current_gamma = GAMMA_START + (GAMMA_END - GAMMA_START) * gamma_progress
        return current_alpha, current_gamma


    def _save_checkpoint(self, episode, is_best=False):
        """Saves the training state to the specific experiment folder."""
        # --- ¡¡MODIFICADO!! Usa la subcarpeta del experimento ---
        if is_best:
            filename = os.path.join(self.experiment_checkpoint_dir, f"qca_best_eps{episode}.pth")
        else:
            filename = os.path.join(self.experiment_checkpoint_dir, f"qca_checkpoint_eps{episode}.pth")
        # -----------------------------------------------------

        if isinstance(self.motor.operator, nn.DataParallel):
            model_to_save = self.motor.operator.module
        else:
            model_to_save = self.motor.operator
        
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod
            
        model_state_dict = model_to_save.state_dict()

        state = {
            'episode': episode,
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
        """Loads the latest training checkpoint from the specific experiment folder."""
        try:
            # --- ¡¡MODIFICADO!! Busca en la subcarpeta del experimento ---
            search_path = os.path.join(self.experiment_checkpoint_dir, "qca_checkpoint_eps*.pth")
            list_of_files = glob.glob(search_path)
            list_of_best_files = glob.glob(os.path.join(self.experiment_checkpoint_dir, "qca_best_eps*.pth"))
            all_checkpoint_files = list_of_files + list_of_best_files

            if not all_checkpoint_files:
                print(f"No checkpoints found in '{self.experiment_checkpoint_dir}'. Starting from scratch.")
                return
            # ---------------------------------------------------------

            latest_file = max(all_checkpoint_files, key=os.path.getmtime)
            
            print(f"Cargando checkpoint: {latest_file}...")
            # CORRECCIÓN: weights_only=False para cargar el optimizador y el historial
            checkpoint = torch.load(latest_file, map_location=DEVICE, weights_only=False)

            target_model = self.motor.operator
            if isinstance(target_model, nn.DataParallel):
                target_model = target_model.module
            if hasattr(target_model, '_orig_mod'):
                 target_model = target_model._orig_mod

            state_dict = checkpoint['model_state_dict']
            is_dataparallel_saved = next(iter(state_dict)).startswith('module.')

            if is_dataparallel_saved:
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                target_model.load_state_dict(new_state_dict, strict=False)
            else:
                target_model.load_state_dict(state_dict, strict=False)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("GradScaler state loaded.")
                
            self.current_episode = checkpoint['episode'] + 1
            self.best_loss = checkpoint['best_loss']
            loaded_history = checkpoint.get('history', {})
            for key in self.history.keys():
                self.history[key] = loaded_history.get(key, [])
            self.stagnation_counter = checkpoint.get('stagnation_counter', 0)
            self.reactivation_counter = checkpoint.get('reactivation_counter', 0)

            print(f"Checkpoint loaded successfully. Resuming from episode {self.current_episode}.")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            self.current_episode = 0
            self.history = {k: [] for k in self.history.keys()}
            self.best_loss = float('inf')
            self.stagnation_counter = 0
            self.reactivation_counter = 0
            self.gradient_norms = []
            self.scaler = GradScaler(device=DEVICE, enabled=(DEVICE.type == 'cuda'))

    # ... (check_stagnation_and_reactivate no cambia) ...
    def check_stagnation_and_reactivate(self, total_episodes):
        current_loss = self.history['Loss'][-1] if self.history['Loss'] else float('inf')
        if not np.isnan(current_loss) and not np.isinf(current_loss) and current_loss < (self.best_loss - MIN_LOSS_IMPROVEMENT):
            self.best_loss = current_loss
            self.stagnation_counter = 0
            return False 
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= STAGNATION_WINDOW:
                print(f"\nSTAGNATION DETECTED at episode {self.current_episode}!")
                print(f"No improvement of {MIN_LOSS_IMPROVEMENT} in {STAGNATION_WINDOW} episodes (or NaN detected).")
                if self.reactivation_counter < REACTIVATION_COUNT:
                    self.reactivation_counter += 1
                    print(f"Attempting reactivation {self.reactivation_counter}/{REACTIVATION_COUNT}...")
                    if REACTIVATION_STATE_MODE == 'random':
                        self.motor.state._reset_state_random()
                        print("-> Resetting state with random noise.")
                    elif REACTIVATION_STATE_MODE == 'seeded':
                         self.motor.state._reset_state_seeded()
                         print("-> Resetting state with central seed.")
                    elif REACTIVATION_STATE_MODE == 'complex_noise':
                         self.motor.state._reset_state_complex_noise()
                         print("-> Resetting state with complex noise.")
                    else:
                         print(f"State reactivation mode '{REACTIVATION_STATE_MODE}' not recognized. Resetting to random.")
                         self.motor.state._reset_state_random()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = current_lr * REACTIVATION_LR_MULTIPLIER
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"-> Learning rate adjusted from {current_lr:.2e} to {new_lr:.2e}.")
                    self.stagnation_counter = 0
                    print("-> Reactivation complete. Continuing training.")
                    return False
                else:
                    print(f"Maximum number of reactivations ({REACTIVATION_COUNT}) reached.")
                    return True
            return False

    def train_episode(self, total_episodes):
        """Runs one full training episode (BPTT-k) with mixed precision."""
        self.motor.state._reset_state_random()
        alpha, gamma = self._calculate_annealed_alpha_gamma(total_episodes)

        episode_total_loss = 0.0
        bptt_cumulative_loss = 0.0
        valid_steps = 0
        current_real = self.motor.state.x_real.clone().requires_grad_(True).to(DEVICE)
        current_imag = self.motor.state.x_imag.clone().requires_grad_(True).to(DEVICE)
        activity_variances_per_step_mean = []
        density_variances_per_step = []
        last_R_density, last_R_causalidad, last_R_stability, last_P_explosion = (float('nan'),)*4
        self.optimizer.zero_grad()

        for t in range(STEPS_PER_EPISODE):
            with autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == 'cuda')):
                if torch.isnan(current_real).any() or torch.isinf(current_real).any():
                    print(f"⚠️  NaN/Inf detected in state at step {t} of episode {self.current_episode}.")
                    episode_total_loss = float('nan')
                    break

                prev_real_detached = current_real.detach()
                prev_imag_detached = current_imag.detach()
                x_cat = torch.cat([current_real.permute(0, 3, 1, 2), current_imag.permute(0, 3, 1, 2)], dim=1)

                if isinstance(self.motor.operator, nn.DataParallel):
                    F_int_real, F_int_imag = self.motor.operator(x_cat)
                else:
                    F_int_real, F_int_imag = self.motor.operator(x_cat)

                if torch.isnan(F_int_real).any() or torch.isinf(F_int_real).any():
                    print(f"⚠️  NaN/Inf detected in F_int at step {t} of episode {self.current_episode}.")
                    episode_total_loss = float('nan')
                    break
                
                bias_real = 0.0
                bias_imag = 0.0
                op_to_check = self.motor.operator.module if isinstance(self.motor.operator, nn.DataParallel) else self.motor.operator
                if hasattr(op_to_check, '_orig_mod'):
                    op_to_check = op_to_check._orig_mod
                if hasattr(op_to_check, 'M_bias_real'):
                    bias_real = op_to_check.M_bias_real.to(DEVICE)
                if hasattr(op_to_check, 'M_bias_imag'):
                    bias_imag = op_to_check.M_bias_imag.to(DEVICE)

                new_real = current_real.squeeze(0) + F_int_real + bias_real
                new_imag = current_imag.squeeze(0) + F_int_imag + bias_imag

                prob_sq = new_real.pow(2) + new_imag.pow(2)
                norm = torch.sqrt(prob_sq.sum(dim=-1, keepdim=True) + 1e-8)
                next_real = new_real / norm
                next_imag = new_imag / norm

                if torch.isnan(next_real).any() or torch.isinf(next_real).any():
                    print(f"⚠️  NaN/Inf detected in next_state at step {t} of episode {self.current_episode}.")
                    episode_total_loss = float('nan')
                    break

                density_map = torch.clamp(prob_sq.sum(dim=-1), 0.0, 3.0)
                current_std_density = density_map.std()
                density_error = torch.abs(current_std_density - TARGET_STD_DENSITY)
                R_density_target = -density_error * (1.0 + density_error)
                change_real = next_real - prev_real_detached.squeeze(0)
                change_imag = next_imag - prev_imag_detached.squeeze(0)
                R_Causalidad = -(change_real.abs().mean() + change_imag.abs().mean())
                density_t_plus_1 = next_real.pow(2) + next_imag.pow(2)
                R_Stability = -density_t_plus_1.var(dim=-1).mean()
                change_magnitude_per_cell = torch.sqrt(change_real.pow(2) + change_imag.pow(2)).sum(dim=-1)
                activity_variances_per_step_mean.append(change_magnitude_per_cell.mean().item())
                density_variances_per_step.append(density_map.var().item())
                P_Explosion = torch.relu(density_map.max() - EXPLOSION_THRESHOLD) * EXPLOSION_PENALTY_MULTIPLIER

                last_R_density, last_R_causalidad, last_R_stability, last_P_explosion = \
                    R_density_target.item(), R_Causalidad.item(), R_Stability.item(), P_Explosion.item()

                reward_step_bptt = (alpha * R_density_target) + \
                                   (BETA_CAUSALITY * R_Causalidad) + \
                                   (gamma * R_Stability) + \
                                   (LAMBDA_ACTIVITY_VAR * change_magnitude_per_cell.var()) - \
                                   (LAMBDA_VELOCIDAD * density_map.var())
                step_loss = -reward_step_bptt + P_Explosion
            
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
                current_real = next_real.unsqueeze(0).detach().to(DEVICE).requires_grad_(True)
                current_imag = next_imag.unsqueeze(0).detach().to(DEVICE).requires_grad_(True)
            
            else: 
                current_real = next_real.unsqueeze(0)
                current_imag = next_imag.unsqueeze(0)

        # --- Fin del Episodio ---
        avg_loss = episode_total_loss / max(valid_steps, 1)
        self.history['Loss'].append(avg_loss)
        self.history['R_Density_Target'].append(last_R_density)
        self.history['R_Causalidad'].append(last_R_causalidad)
        self.history['R_Stability'].append(last_R_stability)
        self.history['P_Explosion'].append(last_P_explosion)
        self.history['R_Activity_Var'].append(np.var(activity_variances_per_step_mean) if len(activity_variances_per_step_mean) > 1 else 0.0)
        self.history['R_Velocidad'].append(np.mean(density_variances_per_step) if density_variances_per_step else 0.0)
        self.history['Gradient_Norm'].append(np.mean(self.gradient_norms) if self.gradient_norms else 0.0)
        self.gradient_norms = []
        self.current_episode += 1
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return avg_loss