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
        """
        Calcula la pérdida como la anti-similitud del coseno entre el estado inicial y final.
        El objetivo es que el estado final sea lo más diferente posible al inicial.
        """
        # Aplanar los tensores para el cálculo de la similitud
        psi_final_flat = psi_final.reshape(psi_final.shape[0], -1)
        psi_inicial_flat = psi_inicial.reshape(psi_inicial.shape[0], -1)

        # Calcular la similitud del coseno. 
        # Se usa la parte real porque el coseno es una medida real.
        cos_sim = torch.nn.functional.cosine_similarity(psi_final_flat.real, psi_inicial_flat.real, dim=1)
        
        # La pérdida es el negativo de la similitud. Queremos maximizar la diferencia.
        # Se toma la media sobre el batch.
        loss = -cos_sim.mean()
        return loss

    def train_episode(self):
        """
        Ejecuta un episodio de entrenamiento completo usando BPTT.
        """
        self.motor.operator.train()
        total_loss = 0.0
        
        # El bucle ahora es sobre episodios, no sobre pasos internos.
        for step in range(self.exp_cfg.STEPS_PER_EPISODE):
            torch.compiler.cudagraph_mark_step_begin()
            self.optimizer_M.zero_grad()

            psi_inicial = self.motor.get_initial_state(self.exp_cfg.BATCH_SIZE_TRAINING)

            with autocast():
                # --- ¡¡REFACTORIZACIÓN BPTT!! ---
                # Propagar una vez y obtener el estado final.
                psi_history, psi_final = self.motor.propagate(psi_inicial, self.exp_cfg.QCA_STEPS_TRAINING)
                loss = self.compute_loss(psi_final, psi_inicial)

            # Retropropagar una sola vez a través de toda la secuencia.
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_M)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / self.exp_cfg.STEPS_PER_EPISODE