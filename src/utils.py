# src/utils.py
import torch
import os
import re

# ¡Importación relativa!
from .qca_engine import Aetheria_Motor

# ------------------------------------------------------------------------------
# 4.2: State Checkpointing Functions
# ------------------------------------------------------------------------------

def load_qca_state(motor_instance: Aetheria_Motor, checkpoint_filepath: str):
    """Loads the QCA state (x_real, x_imag) from a checkpoint file."""
    try:
        device = motor_instance.state.x_real.device
        checkpoint = torch.load(checkpoint_filepath, map_location=device)
        if 'x_real' in checkpoint and 'x_imag' in checkpoint and \
           checkpoint['x_real'].shape == motor_instance.state.x_real.shape and \
           checkpoint['x_imag'].shape == motor_instance.state.x_imag.shape:

            motor_instance.state.x_real.data = checkpoint['x_real'].data.to(device)
            motor_instance.state.x_imag.data = checkpoint['x_imag'].data.to(device)
            print(f"✅ State loaded successfully from: {checkpoint_filepath}")
            return checkpoint.get('step', -1)
        else:
            print(f"❌ Error loading state: Checkpoint file invalid or dimensions mismatch.")
            return -1
    except FileNotFoundError:
        print(f"❌ Error loading state: File '{checkpoint_filepath}' not found.")
        return -1
    except Exception as e:
        print(f"❌ Error loading state from '{checkpoint_filepath}': {e}")
        return -1

def save_qca_state(motor_instance: Aetheria_Motor, step: int, checkpoint_dir: str):
    """Saves the current QCA state (x_real, x_imag) and step number."""
    checkpoint_filename = os.path.join(
        checkpoint_dir,
        f"large_sim_state_step_{step}.pth"
    )
    try:
        torch.save({
            'step': step,
            'x_real': motor_instance.state.x_real.data.cpu(),
            'x_imag': motor_instance.state.x_imag.data.cpu()
        }, checkpoint_filename)
        print(f"\n💾 Large simulation checkpoint saved to: {checkpoint_filename}")
    except Exception as e:
        print(f"⚠️  Error saving large simulation checkpoint at step {step}: {e}")