# src/pipeline_viz.py
import torch
import imageio.v2 as imageio
import os
import time
import glob

# ¡Importaciones relativas!
from . import config as cfg
from .qca_engine import QCA_State, Aetheria_Motor
from .qca_operator_mlp import QCA_Operator_MLP
from .visualization import (
    get_density_frame_gpu, get_channel_frame_gpu,
    get_state_magnitude_frame_gpu, get_state_phase_frame_gpu,
    get_state_change_magnitude_frame_gpu
)

def run_visualization_pipeline(Aetheria_Motor_Train: Aetheria_Motor | None, M_FILENAME: str | None):
    """
    Ejecuta la FASE 6: Visualización Post-Entrenamiento.
    Guarda los videos en el directorio /output.
    """
    print("\n" + "="*60)
    print(">>> INICIANDO FASE DE VISUALIZACIÓN POST-ENTRENAMIENTO (FASE 6) <<<")
    print("="*60)

    # Si no se entrenó, intentar cargar el motor y el modelo
    if Aetheria_Motor_Train is None:
        print("Motor de entrenamiento no proporcionado. Inicializando uno nuevo para visualización...")
        model_M = QCA_Operator_MLP(cfg.D_STATE, cfg.HIDDEN_CHANNELS)
        Aetheria_Motor_Train = Aetheria_Motor(cfg.GRID_SIZE_TRAINING, cfg.D_STATE, model_M)
        
        if not M_FILENAME: # Buscar el último modelo si no se pasó
            model_files = glob.glob(os.path.join(cfg.CHECKPOINT_DIR, f"PEF_Deep_v3_G{cfg.GRID_SIZE_TRAINING}_Eps*_FINAL.pth"))
            M_FILENAME = max(model_files, key=os.path.getctime, default=None) if model_files else None

        if M_FILENAME and os.path.exists(M_FILENAME):
            print(f"Cargando pesos desde: {M_FILENAME}")
            try:
                model_state_dict = torch.load(M_FILENAME, map_location=cfg.DEVICE)
                
                is_dataparallel_saved = next(iter(model_state_dict)).startswith('module.')
                is_dataparallel_current = isinstance(Aetheria_Motor_Train.operator, torch.nn.DataParallel)

                if is_dataparallel_saved and not is_dataparallel_current:
                    new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
                    Aetheria_Motor_Train.operator.load_state_dict(new_state_dict)
                elif not is_dataparallel_saved and is_dataparallel_current:
                    Aetheria_Motor_Train.operator.module.load_state_dict(model_state_dict)
                else:
                    Aetheria_Motor_Train.operator.load_state_dict(model_state_dict)
                
                Aetheria_Motor_Train.operator.eval()
                print("✅ Pesos cargados exitosamente.")
            except Exception as e:
                print(f"❌ Error cargando pesos del modelo '{M_FILENAME}': {e}")
                print("⚠️  La visualización se ejecutará con pesos aleatorios.")
        else:
            print("❌ No se encontró ningún modelo entrenado. La visualización se ejecutará con pesos aleatorios.")

    print(f"Generando {cfg.NUM_FRAMES_VIZ} frames con Ley-M en cuadrícula {cfg.GRID_SIZE_TRAINING}x{cfg.GRID_SIZE_TRAINING}...")

    Aetheria_Motor_Train.operator.eval()
    Aetheria_Motor_Train.state._reset_state_random() 

    FRAMES_DENSITY = []
    FRAMES_CHANNELS = []
    FRAMES_MAGNITUDE = []
    FRAMES_PHASE = []
    FRAMES_CHANGE = []

    prev_state_viz = QCA_State(Aetheria_Motor_Train.size, Aetheria_Motor_Train.d_state)
    prev_state_viz.x_real.data = Aetheria_Motor_Train.state.x_real.data.clone()
    prev_state_viz.x_imag.data = Aetheria_Motor_Train.state.x_imag.data.clone()

    with torch.no_grad():
        for t in range(cfg.NUM_FRAMES_VIZ):
            current_state_clone = QCA_State(Aetheria_Motor_Train.state.size, Aetheria_Motor_Train.state.d_state)
            current_state_clone.x_real.data = Aetheria_Motor_Train.state.x_real.data.clone().to(cfg.DEVICE)
            current_state_clone.x_imag.data = Aetheria_Motor_Train.state.x_imag.data.clone().to(cfg.DEVICE)

            Aetheria_Motor_Train.evolve_step()
            next_state = Aetheria_Motor_Train.state

            FRAMES_DENSITY.append(get_density_frame_gpu(next_state))
            FRAMES_CHANNELS.append(get_channel_frame_gpu(next_state, num_channels=min(3, cfg.D_STATE)))
            FRAMES_MAGNITUDE.append(get_state_magnitude_frame_gpu(next_state))
            FRAMES_PHASE.append(get_state_phase_frame_gpu(next_state))
            FRAMES_CHANGE.append(get_state_change_magnitude_frame_gpu(next_state, prev_state_viz))
            
            prev_state_viz = current_state_clone

            if (t + 1) % (cfg.NUM_FRAMES_VIZ // 10 or 1) == 0:
                print(f"-> Capturando frame {t+1}/{cfg.NUM_FRAMES_VIZ}...")

    print("✅ Captura de frames de visualización completada.")
    print("\n--- GUARDANDO VIDEOS DE VISUALIZACIÓN (Tamaño Entrenamiento) ---")

    try:
        if M_FILENAME:
            BASE_NAME = os.path.basename(M_FILENAME).replace('_FINAL.pth', '')
        else:
            BASE_NAME = f"Viz_TrainSize_G{cfg.GRID_SIZE_TRAINING}_{int(time.time())}"

        # Guardar videos en la carpeta /output principal
        path_density = os.path.join(cfg.OUTPUT_DIR, f"{BASE_NAME}_1_DENSITY.mp4")
        path_channels = os.path.join(cfg.OUTPUT_DIR, f"{BASE_NAME}_2_CHANNELS.mp4")
        path_magnitude = os.path.join(cfg.OUTPUT_DIR, f"{BASE_NAME}_3_MAGNITUDE.mp4")
        path_phase = os.path.join(cfg.OUTPUT_DIR, f"{BASE_NAME}_4_PHASE.mp4")
        path_change = os.path.join(cfg.OUTPUT_DIR, f"{BASE_NAME}_5_CHANGE.mp4")

        imageio.mimsave(path_density, FRAMES_DENSITY, fps=cfg.FPS_VIZ_TRAINING, codec='libx264', quality=8)
        imageio.mimsave(path_channels, FRAMES_CHANNELS, fps=cfg.FPS_VIZ_TRAINING, codec='libx264', quality=8)
        imageio.mimsave(path_magnitude, FRAMES_MAGNITUDE, fps=cfg.FPS_VIZ_TRAINING, codec='libx264', quality=8)
        imageio.mimsave(path_phase, FRAMES_PHASE, fps=cfg.FPS_VIZ_TRAINING, codec='libx264', quality=8)
        imageio.mimsave(path_change, FRAMES_CHANGE, fps=cfg.FPS_VIZ_TRAINING, codec='libx264', quality=8)

        print(f"✅ Videos de visualización guardados en '{cfg.OUTPUT_DIR}':")
        print(f"   -> {os.path.basename(path_density)}")
        print(f"   -> {os.path.basename(path_channels)}")
        print(f"   -> {os.path.basename(path_magnitude)}")
        print(f"   -> {os.path.basename(path_phase)}")
        print(f"   -> {os.path.basename(path_change)}")

    except Exception as e:
        print(f"❌ Error guardando videos de visualización: {e}")

    print("\n>>> FASE DE VISUALIZACIÓN POST-ENTRENAMIENTO (FASE 6) COMPLETADA <<<")