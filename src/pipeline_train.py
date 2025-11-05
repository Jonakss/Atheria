# src/pipeline_train.py
import torch
import torch.nn as nn
import time
import os
import numpy as np

# ¡Importaciones relativas!
from . import config as cfg
from .trainer import QC_Trainer_v3
from .qca_engine import Aetheria_Motor # El motor siempre se importa

# --- CAMBIO AQUÍ: Selector de Modelo (Ley M) ---
# Descomenta la línea del modelo que quieres entrenar.

# Opción 1: El MLP 1x1 original (Rápido, pero "míope")
from .qca_operator_mlp import QCA_Operator_MLP as ActiveModel

# Opción 2: La U-Net (Más lenta, pero con "conciencia regional")
# from .qca_operator_unet import QCA_Operator_UNet as ActiveModel
# -----------------------------------------------


def run_training_pipeline():
    """
    Ejecuta la FASE 5: Lógica Principal de Entrenamiento.
    Retorna el motor entrenado y la ruta al archivo del modelo final.
    """
    print("\n" + "="*60)
    print(">>> INICIANDO FASE DE ENTRENAMIENTO (FASE 5) <<<")
    print(f"Modelo Activo: {ActiveModel.__name__}")
    print("="*60)
    
    model_id = ActiveModel.__name__ # Usar el nombre de la clase
    
    # 1. Crear el modelo
    model_M = ActiveModel(cfg.D_STATE, cfg.HIDDEN_CHANNELS)
    

    # 2. Crear el motor con el modelo (genérico)
    Aetheria_Motor_Train = Aetheria_Motor(cfg.GRID_SIZE_TRAINING, cfg.D_STATE, model_M)
    print(f"Motor y Ley-M ({model_id}) inicializados. Cuadrícula: {cfg.GRID_SIZE_TRAINING}x{cfg.GRID_SIZE_TRAINING}.")

    trainable_params = sum(p.numel() for p in (Aetheria_Motor_Train.operator.module.parameters()
                                                 if isinstance(Aetheria_Motor_Train.operator, nn.DataParallel)
                                                 else Aetheria_Motor_Train.operator.parameters()) if p.requires_grad)
    print(f"Parámetros Entrenables: {trainable_params}")

    trainer = QC_Trainer_v3(Aetheria_Motor_Train, cfg.LR_RATE_M)

    if cfg.CONTINUE_TRAINING:
        print("Intentando continuar entrenamiento...")
        trainer._load_checkpoint()
    else:
        print("Iniciando nuevo entrenamiento desde cero.")

    print(f"Directorio de Checkpoints: {cfg.CHECKPOINT_DIR}")
    print(f"Iniciando desde episodio {trainer.current_episode}. Entrenando por {cfg.EPISODES_TO_ADD} episodios más.")

    start_time = time.time()
    final_episode = trainer.current_episode + cfg.EPISODES_TO_ADD

    try:
        for episode in range(trainer.current_episode, final_episode):
            avg_loss = trainer.train_episode(final_episode)

            if np.isnan(avg_loss) or np.isinf(avg_loss):
                print(f"⚠️  Episodio {episode:04}: Entrenamiento fallido (NaN/Inf).")
            
            if episode % 10 == 0 or episode == final_episode - 1:
                alpha, gamma = trainer._calculate_annealed_alpha_gamma(final_episode)
                last_r_density = trainer.history['R_Density_Target'][-1] if trainer.history['R_Density_Target'] else float('nan')
                print(f"Eps {episode:04}: Loss={avg_loss:.3e} | R_Dens={last_r_density:.3f} | α={alpha:.2f}, γ={gamma:.2f}, LR={trainer.optimizer.param_groups[0]['lr']:.2e}")

            if episode % cfg.SAVE_EVERY_EPISODES == 0 and episode > trainer.current_episode:
                trainer._save_checkpoint(episode)
                if not np.isnan(avg_loss) and not np.isinf(avg_loss) and avg_loss < trainer.best_loss:
                    trainer._save_checkpoint(episode, is_best=True)
                    print(f"🏆 Nuevo mejor modelo guardado en episodio {episode}")

            if trainer.check_stagnation_and_reactivate(final_episode):
                print("Entrenamiento detenido por estancamiento.")
                break

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido. Guardando checkpoint...")
        if trainer: trainer._save_checkpoint(trainer.current_episode)
    except Exception as e:
        print(f"\n❌ Error crítico en entrenamiento: {e}")
        if trainer: trainer._save_checkpoint(trainer.current_episode)
        raise e 

    end_time = time.time()
    print(f"\nEntrenamiento completado en {end_time - start_time:.2f}s.")
    
    # --- GUARDAR MODELO FINAL (Ley M) ---
    M_FILENAME = None
    if trainer:
        TIMESTAMP = int(time.time())
        # Guarda en el directorio de checkpoints configurado
        M_FILENAME = os.path.join(
            cfg.CHECKPOINT_DIR, 
            f"{model_id}_G{cfg.GRID_SIZE_TRAINING}_Eps{trainer.current_episode}_{TIMESTAMP}_FINAL.pth"
        )
        try:
            # Lógica para guardar el estado del modelo (manejando DP y compile)
            model_to_save = trainer.motor.operator
            if isinstance(model_to_save, nn.DataParallel):
                model_to_save = model_to_save.module
            if hasattr(model_to_save, '_orig_mod'):
                model_to_save = model_to_save._orig_mod
            
            model_state_dict_to_save = model_to_save.state_dict()
            torch.save(model_state_dict_to_save, M_FILENAME)
            print(f"✅ Ley Fundamental Final (M) guardada en: {M_FILENAME}")
        except Exception as e:
            print(f"❌ Error guardando modelo final: {e}")

    print(">>> FASE DE ENTRENAMIENTO (FASE 5) COMPLETADA <<<")
    return Aetheria_Motor_Train, M_FILENAME