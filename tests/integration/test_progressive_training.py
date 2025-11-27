import sys
import os
from pathlib import Path
import torch
import shutil
from types import SimpleNamespace
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from src.trainers.qc_trainer_v4 import QC_Trainer_v4
from src.model_loader import instantiate_model, load_weights, load_checkpoint_data

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MOCK CONFIGURATION FOR LOCAL TEST ---
TEST_ROOT_NAME = "Local_Test_Experiment"
OUTPUT_DIR = PROJECT_ROOT / "output" / "test_output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / TEST_ROOT_NAME
LOG_DIR = OUTPUT_DIR / "logs" / TEST_ROOT_NAME
EXPORTS_DIR = OUTPUT_DIR / "exports"

# Clean previous test run
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Mock Docs Directory for Experiment Logger
MOCK_DOCS_DIR = PROJECT_ROOT / "docs" / "40_Experiments" / TEST_ROOT_NAME
if MOCK_DOCS_DIR.exists():
    shutil.rmtree(MOCK_DOCS_DIR)
MOCK_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Define minimal phases for testing
TRAINING_PHASES = [
    {
        "PHASE_NAME": "Phase1_Test",
        "LOAD_FROM_PHASE": None,
        "MODEL_ARCHITECTURE": "UNET",
        "MODEL_PARAMS": {
            "d_state": 4,
            "hidden_channels": 8, # Small model
        },
        "GRID_SIZE_TRAINING": 16, # Small grid
        "QCA_STEPS_TRAINING": 5,
        "LR_RATE_M": 1e-3,
        "GAMMA_DECAY": 0.0,
        "TOTAL_EPISODES": 2, # Only 2 episodes
        "SAVE_EVERY_EPISODES": 1,
    },
    {
        "PHASE_NAME": "Phase2_Test",
        "LOAD_FROM_PHASE": "Phase1_Test",
        "MODEL_ARCHITECTURE": "UNET",
        "MODEL_PARAMS": {
            "d_state": 4,
            "hidden_channels": 16, # Architecture change (transfer learning test)
        },
        "GRID_SIZE_TRAINING": 16,
        "QCA_STEPS_TRAINING": 5,
        "LR_RATE_M": 1e-3,
        "GAMMA_DECAY": 0.0,
        "TOTAL_EPISODES": 2,
        "SAVE_EVERY_EPISODES": 1,
    }
]

GLOBAL_CONFIG = {
    "DRIVE_SYNC_EVERY": 1,
    "MAX_TRAINING_HOURS": 1,
    "AUTO_RESUME": True,
    "MAX_CHECKPOINTS_TO_KEEP": 2,
}

device = torch.device("cpu") # Force CPU for test

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None
    last_model = checkpoint_dir / "last_model.pth"
    if last_model.exists():
        return str(last_model)
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)

def sync_checkpoint_to_drive(local_path, drive_dir, filename=None):
    # Mock drive sync by copying to a 'drive' folder in test output
    try:
        if filename is None:
            filename = Path(local_path).name
        drive_path = drive_dir / filename
        shutil.copy2(local_path, drive_path)
        print(f"💾 [MOCK DRIVE] Synced: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error syncing: {e}")
        return False

def run_test():
    print("🚀 STARTING LOCAL SMOKE TEST")
    print(f"Output Dir: {OUTPUT_DIR}")
    
    # --- TRAINING LOOP (Copied & Adapted from Notebook) ---
    for phase_idx, phase_cfg in enumerate(TRAINING_PHASES):
        PHASE_NAME = phase_cfg["PHASE_NAME"]
        print(f"\n🔹 PHASE {phase_idx+1}: {PHASE_NAME}")
        
        # 1. Directories
        PHASE_CHECKPOINT_DIR = CHECKPOINT_DIR / PHASE_NAME
        PHASE_LOG_DIR = LOG_DIR / PHASE_NAME
        PHASE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        PHASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        LOCAL_PHASE_DIR = OUTPUT_DIR / "local_checkpoints" / TEST_ROOT_NAME / PHASE_NAME
        LOCAL_PHASE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 3. Config
        current_exp_cfg = SimpleNamespace(**phase_cfg)
        current_exp_cfg.MODEL_PARAMS = SimpleNamespace(**phase_cfg["MODEL_PARAMS"])
        current_exp_cfg.DEVICE = device
        
        # 4. Instantiate
        print(f"🛠️ Instantiating model...")
        model = instantiate_model(current_exp_cfg)
        
        # 5. Load Weights / Transfer Learning
        weights_loaded = False
        if phase_cfg["LOAD_FROM_PHASE"]:
            prev_phase_name = phase_cfg["LOAD_FROM_PHASE"]
            print(f"📥 Loading from previous phase: {prev_phase_name}")
            prev_dir = CHECKPOINT_DIR / prev_phase_name
            best_prev = prev_dir / "best_model.pth"
            
            if not best_prev.exists():
                 # Try to find any checkpoint if best not found (for test robustness)
                 best_prev = find_latest_checkpoint(prev_dir)
                 if best_prev: best_prev = Path(best_prev)

            if best_prev and best_prev.exists():
                print(f"✅ Found previous checkpoint: {best_prev.name}")
                ckpt_data = load_checkpoint_data(str(best_prev))
                
                # Smart Load: Filter out size mismatches
                model_state = model.state_dict()
                pretrained_state = ckpt_data['model_state_dict']
                filtered_state = {}
                ignored_keys = []
                
                for k, v in pretrained_state.items():
                    if k in model_state:
                        if v.shape == model_state[k].shape:
                            filtered_state[k] = v
                        else:
                            ignored_keys.append(k)
                    else:
                        # Key not in model (unexpected), ignore or let strict=False handle (but we are filtering manually now)
                        pass
                
                if ignored_keys:
                    print(f"⚠️ Ignoring {len(ignored_keys)} layers due to shape mismatch (Transfer Learning).")
                
                missing, unexpected = model.load_state_dict(filtered_state, strict=False)
                print(f"ℹ️ Transfer Learning: {len(missing)} missing, {len(unexpected)} unexpected keys.")
                weights_loaded = True
            else:
                print(f"⚠️ Previous checkpoint not found at {prev_dir}")
        
        if not weights_loaded:
            print("🆕 Starting from scratch.")
            
        # 6. Trainer
        trainer = QC_Trainer_v4(
            experiment_name=f"{TEST_ROOT_NAME}/{PHASE_NAME}",
            model=model,
            model_params=phase_cfg["MODEL_PARAMS"], # Fix: Pass model params so d_state is correct
            device=device,
            lr=phase_cfg["LR_RATE_M"],
            grid_size=phase_cfg["GRID_SIZE_TRAINING"],
            qca_steps=phase_cfg["QCA_STEPS_TRAINING"],
            gamma_decay=phase_cfg["GAMMA_DECAY"],
            max_checkpoints_to_keep=GLOBAL_CONFIG["MAX_CHECKPOINTS_TO_KEEP"]
        )
        trainer.checkpoint_dir = str(LOCAL_PHASE_DIR)
        
        # 7. Episode Loop
        print(f"▶️ Training {phase_cfg['TOTAL_EPISODES']} episodes...")
        for episode in range(phase_cfg["TOTAL_EPISODES"]):
            loss, metrics = trainer.train_episode(episode)
            # loss = epoch_result.get("loss_total", 0)
            print(f"   Ep {episode}: Loss {loss:.4f}")
            
            # Save/Sync
            if (episode + 1) % phase_cfg["SAVE_EVERY_EPISODES"] == 0:
                trainer.save_checkpoint(episode, is_best=True) # Always save best for test
                sync_checkpoint_to_drive(str(LOCAL_PHASE_DIR / "best_model.pth"), PHASE_CHECKPOINT_DIR)
                
        print(f"✅ Phase {PHASE_NAME} Completed")

    # --- EXPORT TEST (Adapted from Notebook) ---
    print("\n📤 TESTING EXPORT LOGIC...")
    last_phase = TRAINING_PHASES[-1]
    last_phase_name = last_phase["PHASE_NAME"]
    LAST_PHASE_CHECKPOINT_DIR = CHECKPOINT_DIR / last_phase_name
    
    best_checkpoint = LAST_PHASE_CHECKPOINT_DIR / "best_model.pth"
    if best_checkpoint.exists():
        print(f"✅ Found final checkpoint: {best_checkpoint}")
        # Mock export
        shutil.copy2(best_checkpoint, EXPORTS_DIR / "final_model_test.pth")
        print("✅ Export successful")
    else:
        print("❌ Final checkpoint NOT found!")
        sys.exit(1)

    print("\n✅✅✅ TEST COMPLETED SUCCESSFULLY ✅✅✅")
    
    # Cleanup Mock Docs
    if MOCK_DOCS_DIR.exists():
        shutil.rmtree(MOCK_DOCS_DIR)
        print(f"🧹 Cleaned up mock docs: {MOCK_DOCS_DIR}")

if __name__ == "__main__":
    run_test()
