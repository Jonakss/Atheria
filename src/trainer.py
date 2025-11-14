# src/trainer.py
import torch
import os
import json
import logging
from types import SimpleNamespace
from .qca_engine import Aetheria_Motor
from .model_loader import load_model_for_training
from .utils import load_experiment_config, save_checkpoint

# Configuración del logging para el script de entrenamiento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_loss(motor, initial_psi, final_psi_real, config):
    psi_history, _ = motor.propagate(initial_psi, config.QCA_STEPS_TRAINING)
    final_psi_computed = psi_history[-1]
    
    loss_dist = torch.mean((final_psi_computed.abs() - final_psi_real.abs()).pow(2))
    
    laplacian = motor.laplacian_2d_psi(final_psi_computed)
    loss_smooth = torch.mean(laplacian.abs().pow(2))
    
    loss = loss_dist + config.MODEL_PARAMS.get('beta', 0.1) * loss_smooth
    return loss

def run_training_loop(experiment_name):
    try:
        config, checkpoint_path, state = load_experiment_config(experiment_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        motor = load_model_for_training(config, state, device)
        
        start_episode = state.get('episode', 0)
        total_episodes = config.TOTAL_EPISODES

        if start_episode >= total_episodes:
            log_msg = {"type": "log", "payload": "El entrenamiento ya ha alcanzado el número total de episodios."}
            print(json.dumps(log_msg), flush=True)
            return

        optimizer = torch.optim.Adam(motor.operator.parameters(), lr=config.LR_RATE_M)
        if 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])

        loss_history = []

        # --- ¡¡CORRECCIÓN!! Bucle sin tqdm ---
        for episode in range(start_episode, total_episodes):
            initial_psi = motor.get_initial_state(batch_size=1)
            final_psi_real = motor.get_initial_state(batch_size=1)
            
            optimizer.zero_grad()
            loss = compute_loss(motor, initial_psi, final_psi_real, config)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            
            # --- ¡¡CORRECCIÓN!! Enviar progreso en CADA episodio ---
            avg_loss = sum(loss_history) / len(loss_history)
            progress_payload = {
                "type": "progress",
                "payload": {
                    "current_episode": episode + 1,
                    "total_episodes": total_episodes,
                    "avg_loss": avg_loss
                }
            }
            print(json.dumps(progress_payload), flush=True)

            if (episode + 1) % 10 == 0:
                loss_history = [] # Resetear historial de pérdida

            if (episode + 1) % config.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(motor, optimizer, episode + 1, experiment_name)

        save_checkpoint(motor, optimizer, total_episodes, experiment_name)
        final_log = {"type": "log", "payload": f"Entrenamiento completado. Modelo guardado en el episodio {total_episodes}."}
        print(json.dumps(final_log), flush=True)

    except Exception as e:
        error_log = {"type": "log", "payload": f"ERROR en el entrenamiento: {e}"}
        print(json.dumps(error_log), flush=True)
        logging.error("Error en el bucle de entrenamiento", exc_info=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()
    run_training_loop(args.experiment_name)