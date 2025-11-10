import argparse
import os
import re
import subprocess
import shutil

def update_config_file(config_path, params):
    with open(config_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        # Update EXPERIMENT_NAME
        if line.strip().startswith("EXPERIMENT_NAME ="):
            updated_lines.append(f'EXPERIMENT_NAME = "{params["experiment_name"]}"\n')
        # Update EPISODES_TO_ADD
        elif line.strip().startswith("EPISODES_TO_ADD ="):
            updated_lines.append(f'EPISODES_TO_ADD = {params["episodes"]}\n')
        # Update LR_RATE_M
        elif line.strip().startswith("LR_RATE_M ="):
            updated_lines.append(f'LR_RATE_M = {params["lr"]}\n')
        # Update GRID_SIZE_TRAINING
        elif line.strip().startswith("GRID_SIZE_TRAINING ="):
            updated_lines.append(f'GRID_SIZE_TRAINING = {params["grid_size"]}\n')
        # Update CONTINUE_TRAINING
        elif line.strip().startswith("CONTINUE_TRAINING ="):
            updated_lines.append(f'CONTINUE_TRAINING = {params["continue_training"]}\n')
        else:
            updated_lines.append(line)
    
    with open(config_path, 'w') as f:
        f.writelines(updated_lines)

def main():
    parser = argparse.ArgumentParser(description="Update config.py and run Aetheria training.")
    parser.add_argument("--experiment_name", type=str, default="Default_Experiment", help="Name for the training experiment.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train for.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the model.")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size for training.")
    parser.add_argument("--continue_training", type=str, default="False", help="Whether to continue training from the last checkpoint (True/False).") # Use str for boolean from shell

    args = parser.parse_args()

    # Convert string boolean to Python boolean
    args.continue_training = args.continue_training.lower() == 'true'

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'src', 'config.py')
    main_script_path = os.path.join(project_root, 'main.py')

    # Make a backup of config.py
    backup_config_path = config_path + '.bak'
    shutil.copyfile(config_path, backup_config_path)
    print(f"Backup of {config_path} created at {backup_config_path}")

    try:
        # Update config.py
        update_config_file(config_path, vars(args))
        print(f"Updated {config_path} with parameters: {vars(args)}")

        # Run main.py
        print(f"Running {main_script_path}...")
        result = subprocess.run(['python3', main_script_path], capture_output=True, text=True, check=False)
        print("--- Training Output ---")
        print(result.stdout)
        print(result.stderr)
        print("--- End Training Output ---")

        if result.returncode != 0:
            print(f"Training script exited with error code {result.returncode}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Restore config.py from backup
        shutil.move(backup_config_path, config_path)
        print(f"Restored {config_path} from backup.")

if __name__ == "__main__":
    main()
