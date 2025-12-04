
import sys
import os
import shutil
import torch
import logging
import json
from types import SimpleNamespace
from src.utils import load_experiment_config, get_latest_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiment(name, params_dict, loss_fn_name=None, load_from=None, episodes=5):
    """
    Helper to run a training session programmatically by simulating CLI args or calling pipeline directly.
    We'll mimic the behavior of src.trainer.main or call run_training_pipeline directly.
    """
    from src.pipelines.pipeline_train import run_training_pipeline
    from src import config as global_cfg

    print(f"\n{'='*60}")
    print(f"🚀 STARTING EXPERIMENT: {name}")
    print(f"   Objective: {loss_fn_name if loss_fn_name else 'Default (Evolutionary)'}")
    if load_from:
        print(f"   Transfer Learning: Loading from {load_from}")
    print(f"{'='*60}\n")

    # Clean up previous run if exists
    ckpt_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, name)
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        logging.info(f"Cleaned up old checkpoints for {name}")

    # Construct Config
    exp_config = {
        "EXPERIMENT_NAME": name,
        "MODEL_ARCHITECTURE": "UNET", # Simple U-Net for demo
        "LR_RATE_M": 0.001,
        "GRID_SIZE_TRAINING": 32, # Small grid for speed
        "QCA_STEPS_TRAINING": 8,
        "TOTAL_EPISODES": episodes,
        "MODEL_PARAMS": SimpleNamespace(**params_dict),
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "GAMMA_DECAY": 0.0, # Closed system for physics phase
        "NOISE_LEVEL": 0.0,
        "ENGINE_TYPE": "PYTHON",
        "BACKEND_TYPE": "LOCAL",
        "SAVE_EVERY_EPISODES": 2,
        "TRAINER_VERSION": "v4"
    }

    if loss_fn_name:
        exp_config["LOSS_FUNCTION"] = loss_fn_name

    if load_from:
        exp_config["LOAD_FROM_EXPERIMENT"] = load_from

    # Convert to Namespace
    exp_cfg = SimpleNamespace(**exp_config)

    # Run Pipeline
    try:
        run_training_pipeline(exp_cfg)
        print(f"\n✅ Experiment {name} completed successfully.")
        return True
    except Exception as e:
        print(f"\n❌ Experiment {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Common Model Params
    model_params = {
        "d_state": 4,
        "hidden_channels": 16
    }

    # --- PHASE 1: PRE-TRAINING (PHYSICS) ---
    # Objective: Minimize Energy Error (Learn Conservation)
    exp_phase1 = "demo_transfer_phase1_physics"
    success_1 = run_experiment(
        name=exp_phase1,
        params_dict=model_params,
        loss_fn_name="mse_energy", # Use the custom loss we implemented
        episodes=5
    )

    if not success_1:
        sys.exit(1)

    # Verify Checkpoint Exists
    ckpt_1 = get_latest_checkpoint(exp_phase1)
    if not ckpt_1:
        print(f"❌ Phase 1 failed to produce a checkpoint.")
        sys.exit(1)
    print(f"💾 Phase 1 Checkpoint: {ckpt_1}")

    # --- PHASE 2: FINE-TUNING (BIOLOGY) ---
    # Objective: Maximize Complexity/Life (Evolutionary Loss)
    # Start from Phase 1 weights
    exp_phase2 = "demo_transfer_phase2_biology"

    # Run Experiment with Transfer Learning
    success_2 = run_experiment(
        name=exp_phase2,
        params_dict=model_params,
        loss_fn_name="evolutionary", # Switch to biological loss
        load_from=exp_phase1,        # Load weights from Phase 1
        episodes=5
    )

    if success_2:
        print("\n🎉 DEMO COMPLETE: Transfer Learning Successful!")
        print("1. Model learned Energy Conservation (MSE Loss)")
        print(f"2. Model transferred weights from '{exp_phase1}'")
        print("3. Model fine-tuned for Complexity (Evolutionary Loss)")
    else:
        print("\n❌ Demo Failed during Phase 2.")
        sys.exit(1)

if __name__ == "__main__":
    main()
