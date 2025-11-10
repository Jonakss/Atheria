# src/pipeline_train.py
import torch
import torch.nn as nn
import time
import os
import numpy as np

# ¡Importaciones relativas!
from . import config as cfg
from .trainer import QC_Trainer_v3
from .qca_engine import Aetheria_Motor 

# src/pipeline_train.py
import torch
import torch.nn as nn
import time
import os
import numpy as np

# ¡Importaciones relativas!
from . import config as cfg
from .trainer import QC_Trainer_v3
from .qca_engine import Aetheria_Motor 
from . import models # <-- ¡NUEVO!

def run_training_pipeline():
    """
    Ejecuta la FASE 5: Lógica Principal de Entrenamiento.
    """
    print("\n" + "="*60)
    print(">>> INICIANDO FASE DE ENTRENAMIENTO (FASE 5) <<<")
    print(f"Nombre del Experimento: {cfg.EXPERIMENT_NAME}")

    # --- Selector de Modelo Dinámico ---
    ActiveModel = models.get_model_class(cfg.ACTIVE_QCA_OPERATOR)
    print(f"Modelo Activo: {ActiveModel.__name__}")
    # -----------------------------------
    
    model_id = ActiveModel.__name__
    
    # --- Instanciación y Compilación del Modelo ---
    # Manejar el caso especial del constructor de UNET_UNITARIA
    if cfg.ACTIVE_QCA_OPERATOR == "UNET_UNITARIA":
        model_M = ActiveModel(d_vector=cfg.D_STATE, hidden_channels=cfg.HIDDEN_CHANNELS)
    else:
        model_M = ActiveModel(d_state=cfg.D_STATE, hidden_channels=cfg.HIDDEN_CHANNELS)

    if cfg.DEVICE.type == 'cuda':
        try:
            print("Aplicando torch.compile() al modelo...")
            # ¡CORREGIDO! Compilar la instancia del modelo, no un entero.
            model_M = torch.compile(model_M, mode="reduce-overhead")
            print("¡torch.compile() aplicado exitosamente!")
        except Exception as e:
            print(f"Advertencia: torch.compile() falló. Se usará el modelo estándar. Error: {e}")

    # --- Inicialización del Motor ---
    # ¡CORREGIDO! Pasar la instancia del modelo y el d_vector correcto.
    Aetheria_Motor_Train = Aetheria_Motor(
        size=cfg.GRID_SIZE_TRAINING, 
        d_vector=cfg.D_STATE * 2, # El motor espera el total de canales (real + imag)
        operator_model=model_M
    )
    print(f"Motor y Ley-M ({model_id}) inicializados. Cuadrícula: {cfg.GRID_SIZE_TRAINING}x{cfg.GRID_SIZE_TRAINING}.")
    # --------------------------------

    trainable_params = sum(p.numel() for p in (Aetheria_Motor_Train.operator.module.parameters()
                                                 if isinstance(Aetheria_Motor_Train.operator, nn.DataParallel)
                                                 else Aetheria_Motor_Train.operator.parameters()) if p.requires_grad)
    print(f"Parámetros Entrenables: {trainable_params}")

    trainer = QC_Trainer_v3(
        Aetheria_Motor_Train, 
        cfg.LR_RATE_M,
        cfg.EXPERIMENT_NAME
    )

    if cfg.CONTINUE_TRAINING:
        print("Intentando continuar entrenamiento...")
        trainer._load_checkpoint()
    else:
        print("Iniciando nuevo entrenamiento desde cero.")

    print(f"Directorio de Checkpoints: {trainer.experiment_checkpoint_dir}")
    print(f"Iniciando desde episodio {trainer.current_episode}. Entrenando por {cfg.EPISODES_TO_ADD} episodios más.")

    start_time = time.time()
    final_episode = trainer.current_episode + cfg.EPISODES_TO_ADD

    try:
        for episode in range(trainer.current_episode, final_episode):
            avg_loss = trainer.train_episode(final_episode)

            if np.isnan(avg_loss) or np.isinf(avg_loss):
                print(f"⚠️  Episodio {episode:04}: Entrenamiento fallido (NaN/Inf).")
            
            # --- ¡¡MODIFICADO!! Imprimir las nuevas recompensas ---
            if episode % 10 == 0 or episode == final_episode - 1:
                last_r_quiet = trainer.history['R_Quietud'][-1] if trainer.history['R_Quietud'] else float('nan')
                last_r_complex = trainer.history['R_Complejidad_Localizada'][-1] if trainer.history['R_Complejidad_Localizada'] else float('nan')
                last_grad_norm = trainer.history['Gradient_Norm'][-1] if trainer.history['Gradient_Norm'] else float('nan')

                print(f"Eps {episode:04}: Loss={avg_loss:.3e} | "
                      f"R_Quietud={last_r_quiet:.3f} (Peso: {cfg.PESO_QUIETUD}) | "
                      f"R_Complex={last_r_complex:.3f} (Peso: {cfg.PESO_COMPLEJIDAD_LOCALIZADA}) | "
                      f"GradNorm={last_grad_norm:.3e} | "
                      f"LR={trainer.optimizer.param_groups[0]['lr']:.2e}")
            # ---------------------------------------------------

            if (episode + 1) % cfg.SAVE_EVERY_EPISODES == 0:
                trainer._save_checkpoint(episode + 1)
                if not np.isnan(avg_loss) and not np.isinf(avg_loss) and avg_loss < trainer.best_loss:
                    trainer._save_checkpoint(episode + 1, is_best=True)
                    print(f"🏆 Nuevo mejor modelo guardado en episodio {episode + 1}")

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
    
    M_FILENAME = None
    if trainer:
        TIMESTAMP = int(time.time())
        M_FILENAME = os.path.join(
            trainer.experiment_checkpoint_dir, 
            f"{model_id}_G{cfg.GRID_SIZE_TRAINING}_Eps{trainer.current_episode}_{TIMESTAMP}_FINAL.pth"
        )
        try:
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