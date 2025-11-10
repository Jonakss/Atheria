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
    DEVICE, CHECKPOINT_DIR, EXPERIMENT_NAME, 
    # --- ¡¡NUEVAS RECOMPENSAS IMPORTADAS!! ---
    PESO_QUIETUD, PESO_COMPLEJIDAD_LOCALIZADA,
    # -----------------------------------------
    STAGNATION_WINDOW, MIN_LOSS_IMPROVEMENT, REACTIVATION_COUNT, 
    REACTIVATION_STATE_MODE, REACTIVATION_LR_MULTIPLIER, 
    GRADIENT_CLIP, STEPS_PER_EPISODE, PERSISTENCE_COUNT,
    STATE_VECTOR_DIM # <--- ¡NUEVO!
)

# ------------------------------------------------------------------------------
# 2.1: QC_Trainer_v3 Class
# ------------------------------------------------------------------------------
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
        self.experiment_checkpoint_dir = os.path.join(CHECKPOINT_DIR, self.experiment_name)
        os.makedirs(self.experiment_checkpoint_dir, exist_ok=True)

        # --- ¡¡MODIFICADO!! Historial de las nuevas recompensas ---
        self.history = {
            'Loss': [],
            'R_Quietud': [],
            'R_Complejidad_Localizada': [],
            'Gradient_Norm': [],
        }
        # -------------------------------------------------------

        self.current_episode = 0
        self.best_loss = float('inf')
        self.stagnation_counter = 0
        self.reactivation_counter = 0
        self.gradient_norms = []

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
            search_path = os.path.join(self.experiment_checkpoint_dir, "qca_checkpoint_eps*.pth")
            list_of_files = glob.glob(search_path)
            list_of_best_files = glob.glob(os.path.join(self.experiment_checkpoint_dir, "qca_best_eps*.pth"))
            all_checkpoint_files = list_of_files + list_of_best_files

            if not all_checkpoint_files:
                print(f"No checkpoints found in '{self.experiment_checkpoint_dir}'. Starting from scratch.")
                return

            latest_file = max(all_checkpoint_files, key=os.path.getmtime)
            
            print(f"Cargando checkpoint: {latest_file}...")
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