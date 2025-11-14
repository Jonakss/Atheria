# src/trainer.py
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import logging
from types import SimpleNamespace
import torch.compiler

# ... (el resto de las importaciones)

class QC_Trainer_v3:
    def __init__(self, motor, lr_m, global_cfg, exp_cfg: SimpleNamespace):
        # ... (el resto del __init__)
        self.motor = motor
        self.device = motor.device
        self.optimizer_M = optim.Adam(self.motor.operator.parameters(), lr=lr_m)
        self.scaler = GradScaler()
        self.global_cfg = global_cfg
        self.exp_cfg = exp_cfg
        self.checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, exp_cfg.EXPERIMENT_NAME)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.current_episode = 0

    def _save_checkpoint(self, episode):
        # ... (código de guardar checkpoint)
        pass

    def _load_checkpoint(self):
        # ... (código de cargar checkpoint)
        pass

    def compute_loss(self, psi_final, psi_inicial):
        # ... (código de calcular pérdida)
        pass

    def train_episode(self):
        self.motor.operator.train()
        total_loss = 0.0
        
        for step in range(self.exp_cfg.STEPS_PER_EPISODE):
            # --- ¡¡SOLUCIÓN DEFINITIVA!! Marcar el inicio del paso para CUDAGraphs ---
            torch.compiler.cudagraph_mark_step_begin()
            
            self.optimizer_M.zero_grad()

            # Generar estado inicial aleatorio
            psi_inicial = self.motor.get_initial_state(self.exp_cfg.BATCH_SIZE_TRAINING)

            with autocast():
                # Propagar el estado a través del motor
                psi_final = self.motor.propagate(psi_inicial, self.exp_cfg.QCA_STEPS_TRAINING)
                
                # Calcular la pérdida
                loss = self.compute_loss(psi_final, psi_inicial)

            # Backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_M)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / self.exp_cfg.STEPS_PER_EPISODE