# src/pipelines/pipeline_train.py
import torch
from types import SimpleNamespace
import logging
import os

from .. import config as global_cfg
from ..model_loader import instantiate_model, load_checkpoint_data, load_weights
from ..engines.qca_engine import Aetheria_Motor
from ..trainers import QC_Trainer_v3, QC_Trainer_v4
from ..utils import get_latest_checkpoint, load_experiment_config, update_simple_namespace

def run_training_pipeline(exp_cfg: SimpleNamespace, checkpoint_path: str | None = None):
    """
    Ejecuta el pipeline de entrenamiento completo para una configuración de experimento dada.
    """
    device = global_cfg.get_device()
    logging.info(f"Pipeline de entrenamiento iniciado en el dispositivo: {device}")

    trainer_version = _determine_trainer_version(exp_cfg)
    logging.info(f"Usando entrenador: {trainer_version}")

    ley_M = None

    if hasattr(exp_cfg, 'LOAD_FROM_EXPERIMENT') and exp_cfg.LOAD_FROM_EXPERIMENT:
        # --- LÓGICA DE TRANSFER LEARNING REFACTORIZADA ---
        logging.info(f"Iniciando Transfer Learning desde: '{exp_cfg.LOAD_FROM_EXPERIMENT}'")

        base_checkpoint_path = get_latest_checkpoint(exp_cfg.LOAD_FROM_EXPERIMENT)
        if not base_checkpoint_path:
            logging.error(f"No se encontró checkpoint para el experimento base. Abortando.")
            return
            
        checkpoint_data = load_checkpoint_data(base_checkpoint_path)
        if not checkpoint_data or 'exp_config' not in checkpoint_data:
            logging.error(f"El checkpoint base no contiene configuración. Abortando.")
            return
            
        # 1. Cargar y fusionar configuraciones
        base_config_dict = checkpoint_data['exp_config']
        base_exp_cfg = SimpleNamespace(**base_config_dict)
        if isinstance(base_config_dict.get("MODEL_PARAMS"), dict):
            base_exp_cfg.MODEL_PARAMS = SimpleNamespace(**base_config_dict["MODEL_PARAMS"])

        # Sobrescribir con la nueva configuración (la del usuario)
        update_simple_namespace(base_exp_cfg, exp_cfg)
        exp_cfg = base_exp_cfg
        logging.info(f"Configuración final para Transfer Learning: {exp_cfg}")

        # 2. Instanciar el modelo con la nueva configuración fusionada
        ley_M = instantiate_model(exp_cfg)

        # 3. Cargar los pesos del checkpoint en el nuevo modelo
        load_weights(ley_M, checkpoint_data)
        ley_M.train()
        logging.info("✅ Transfer learning completado: modelo instanciado con nueva config y pesos cargados.")

    elif checkpoint_path:
        # --- LÓGICA DE CONTINUAR ENTRENAMIENTO REFACTORIZADA ---
        logging.info(f"Continuando entrenamiento desde: {checkpoint_path}")

        checkpoint_data = load_checkpoint_data(checkpoint_path)
        if not checkpoint_data:
            return

        # Si el checkpoint tiene config, usarla
        if 'exp_config' in checkpoint_data:
            loaded_config_dict = checkpoint_data['exp_config']
            exp_cfg = SimpleNamespace(**loaded_config_dict)
            if isinstance(loaded_config_dict.get("MODEL_PARAMS"), dict):
                exp_cfg.MODEL_PARAMS = SimpleNamespace(**loaded_config_dict["MODEL_PARAMS"])
            logging.info("Configuración del experimento actualizada desde el checkpoint.")

        # Instanciar modelo y cargar pesos
        ley_M = instantiate_model(exp_cfg)
        load_weights(ley_M, checkpoint_data)
        ley_M.train()
        logging.info("Modelo cargado desde checkpoint.")

    elif trainer_version == "v3":
        # V3 necesita el modelo creado explícitamente
        ley_M = instantiate_model(exp_cfg)
        ley_M.train()
        logging.info("Nuevo modelo creado para entrenamiento (V3).")

    # 2. Inicializar el Entrenador según la versión
    if trainer_version == "v4":
        # QC_Trainer_v4 tiene una interfaz diferente
        # Si ya tenemos un modelo cargado (checkpoint/transfer), pasarlo directamente
        # Si no, crear uno nuevo pasando model_class y model_params
        
        # Convertir MODEL_PARAMS a dict si es necesario
        if isinstance(exp_cfg.MODEL_PARAMS, SimpleNamespace):
            model_params_dict = vars(exp_cfg.MODEL_PARAMS)
        elif isinstance(exp_cfg.MODEL_PARAMS, dict):
            model_params_dict = exp_cfg.MODEL_PARAMS
        else:
            # Intentar acceder como atributo
            model_params_dict = {k: getattr(exp_cfg.MODEL_PARAMS, k) for k in dir(exp_cfg.MODEL_PARAMS) if not k.startswith('_')}
        
        # Si ya tenemos un modelo cargado (checkpoint o transfer learning), usarlo
        # Si no, V4 creará su propio modelo
        trainer_kwargs = {
            'experiment_name': exp_cfg.EXPERIMENT_NAME,
            'device': device,
            'lr': exp_cfg.LR_RATE_M,
            'grid_size': exp_cfg.GRID_SIZE_TRAINING,
            'qca_steps': exp_cfg.QCA_STEPS_TRAINING,
            'gamma_decay': getattr(exp_cfg, 'GAMMA_DECAY', getattr(global_cfg, 'GAMMA_DECAY', 0.01)),
            'max_noise': getattr(exp_cfg, 'NOISE_LEVEL', 0.05)
        }
        
        # Si ya creamos el modelo (checkpoint o transfer), pasarlo
        if 'ley_M' in locals() and ley_M is not None:
            trainer_kwargs['model'] = ley_M
            trainer_kwargs['model_params'] = model_params_dict  # Para obtener d_state
        else:
            # Crear nuevo modelo - necesitamos la clase del modelo
            from ..model_loader import MODEL_MAP
            if exp_cfg.MODEL_ARCHITECTURE not in MODEL_MAP:
                logging.error(f"Arquitectura de modelo desconocida: '{exp_cfg.MODEL_ARCHITECTURE}'")
                raise ValueError(f"Arquitectura de modelo desconocida: '{exp_cfg.MODEL_ARCHITECTURE}'")
            trainer_kwargs['model_class'] = MODEL_MAP[exp_cfg.MODEL_ARCHITECTURE]
            trainer_kwargs['model_params'] = model_params_dict
        
        trainer = QC_Trainer_v4(**trainer_kwargs)
        
        # Log de parámetros
        num_params = sum(p.numel() for p in trainer.motor.operator.parameters() if p.requires_grad)
        print(f"Motor y Ley-M ({exp_cfg.MODEL_ARCHITECTURE}) inicializados (V4). Cuadrícula: {exp_cfg.GRID_SIZE_TRAINING}x{exp_cfg.GRID_SIZE_TRAINING}.")
        print(f"Parámetros Entrenables: {num_params}")
        
        # V4 tiene un método diferente: train_episode(episode_num) que retorna (loss, metrics)
        _run_v4_training_loop(trainer, exp_cfg)
    else:
        # QC_Trainer_v3 usa la interfaz tradicional - necesita el motor creado
        # 2. Inicializar el Motor Aetheria
        # Pasar exp_cfg para que el motor tenga acceso a GAMMA_DECAY (término Lindbladian)
        motor = Aetheria_Motor(ley_M, exp_cfg.GRID_SIZE_TRAINING, exp_cfg.MODEL_PARAMS.d_state, device, cfg=exp_cfg)

        # 3. Compilar el modelo si es compatible
        if getattr(ley_M, '_compiles', True):
            motor.compile_model()
        else:
            logging.info(f"El modelo '{exp_cfg.MODEL_ARCHITECTURE}' ha deshabilitado torch.compile(). Se ejecutará sin compilar.")
        
        # Usar get_model_for_params() para acceder a los parámetros correctamente
        # (maneja el caso cuando el modelo está compilado)
        model_for_params = motor.get_model_for_params()
        num_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
        print(f"Motor y Ley-M ({exp_cfg.MODEL_ARCHITECTURE}) inicializados (V3). Cuadrícula: {exp_cfg.GRID_SIZE_TRAINING}x{exp_cfg.GRID_SIZE_TRAINING}.")
        print(f"Parámetros Entrenables: {num_params}")

        # Inicializar el Entrenador V3
        trainer = QC_Trainer_v3(motor, exp_cfg.LR_RATE_M, global_cfg, exp_cfg)
        trainer.run_training_loop()

    logging.info("Pipeline de entrenamiento finalizado.")


def _determine_trainer_version(exp_cfg: SimpleNamespace) -> str:
    """
    Determina qué versión de entrenador usar basándose en la configuración.
    
    Lógica:
    - Si TRAINER_VERSION está especificada explícitamente en exp_cfg, usarla.
    - Si el experimento ya existe y tiene checkpoints V3, usar V3 (compatibilidad).
    - Si el experimento es nuevo, usar V4 por defecto (Atheria 4).
    
    Returns:
        "v3" o "v4"
    """
    # 1. Verificar si está explícitamente especificado
    if hasattr(exp_cfg, 'TRAINER_VERSION'):
        version = getattr(exp_cfg, 'TRAINER_VERSION', '').lower()
        if version in ['v3', 'v4']:
            logging.info(f"Versión de entrenador especificada explícitamente: {version}")
            return version
    
    # 2. Verificar si el experimento ya existe
    exp_config = load_experiment_config(exp_cfg.EXPERIMENT_NAME)
    
    # 3. Si existe, verificar la versión en su configuración guardada
    if exp_config:
        if hasattr(exp_config, 'TRAINER_VERSION'):
            saved_version = getattr(exp_config, 'TRAINER_VERSION', '').lower()
            if saved_version in ['v3', 'v4']:
                logging.info(f"Versión de entrenador detectada desde config guardada: {saved_version}")
                return saved_version
        
        # Si hay checkpoints, asumir V3 por defecto (compatibilidad hacia atrás)
        checkpoint_path = get_latest_checkpoint(exp_cfg.EXPERIMENT_NAME)
        if checkpoint_path:
            logging.info(f"Checkpoint existente detectado. Usando V3 por compatibilidad.")
            return "v3"
    
    # 4. Por defecto: V4 para nuevos experimentos (Atheria 4)
    logging.info("Experimento nuevo. Usando V4 por defecto (Atheria 4).")
    return "v4"


def _run_v4_training_loop(trainer: QC_Trainer_v4, exp_cfg: SimpleNamespace):
    """
    Ejecuta el bucle de entrenamiento para QC_Trainer_v4.
    V4 tiene una interfaz diferente que requiere adaptación.
    """
    import time
    from datetime import datetime
    from ..utils import save_experiment_config
    
    total_episodes = exp_cfg.TOTAL_EPISODES
    start_episode = getattr(exp_cfg, 'START_EPISODE', 0)
    save_every = getattr(exp_cfg, 'SAVE_EVERY_EPISODES', 50)
    
    logging.info(f"Iniciando bucle de entrenamiento V4. Episodios: {start_episode} a {total_episodes}")
    training_start_time = time.time()
    
    # Sistema Smart Save: Seguimiento del mejor checkpoint
    best_combined_metric = float('inf')  # Menor es mejor
    last_loss = 0.0
    last_metrics = {}
    
    try:
        for episode in range(start_episode, total_episodes):
            try:
                loss, metrics = trainer.train_episode(episode)
                last_loss = loss
                last_metrics = metrics
                
                # Calcular métrica combinada para determinar si es el mejor
                survival = metrics.get('survival', 0.0)
                symmetry = metrics.get('symmetry', 0.0)
                combined_metric = (10.0 * survival) + (5.0 * symmetry)  # Misma ponderación que en loss_function_evolutionary
                
                # Determinar si este es el mejor checkpoint hasta ahora
                is_best = combined_metric < best_combined_metric
                if is_best:
                    best_combined_metric = combined_metric
                
                # Log cada 10 episodios
                if episode % 10 == 0:
                    best_marker = " ⭐ BEST" if is_best else ""
                    logging.info(
                        f"Episodio {episode}/{total_episodes} | "
                        f"Loss: {loss:.6f} | "
                        f"Survival: {survival:.6f} | Symmetry: {symmetry:.6f} | "
                        f"Complexity: {metrics.get('complexity', 0):.6f} | "
                        f"Combined: {combined_metric:.6f}{best_marker}"
                    )
                    print(
                        f"Episodio {episode}/{total_episodes} | "
                        f"Loss: {loss:.6f} | "
                        f"Survival: {survival:.6f} | Symmetry: {symmetry:.6f} | "
                        f"Complexity: {metrics.get('complexity', 0):.6f} | "
                        f"Combined: {combined_metric:.6f}{best_marker}",
                        flush=True
                    )
                
                # Guardar checkpoint periódicamente
                if (episode + 1) % save_every == 0:
                    trainer.save_checkpoint(
                        episode,
                        current_metrics=metrics,
                        current_loss=loss,
                        is_best=is_best
                    )
                
                # Limpiar caché CUDA periódicamente para evitar OutOfMemoryError
                # Limpiar cada 20 episodios o después de guardar checkpoint
                if (episode + 1) % 20 == 0 or (episode + 1) % save_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        logging.debug(f"Memoria CUDA limpiada después del episodio {episode + 1}")
                    
            except torch.cuda.OutOfMemoryError as e:
                logging.error(f"❌ CUDA Out of Memory en episodio {episode}: {e}")
                # Limpiar caché CUDA y reintentar una vez
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    logging.warning(f"⚠️ Memoria CUDA limpiada. Intentando continuar...")
                    try:
                        # Reintentar el episodio
                        loss, metrics = trainer.train_episode(episode)
                        last_loss = loss
                        last_metrics = metrics
                        logging.info(f"✅ Episodio {episode} completado después de limpiar memoria")
                    except torch.cuda.OutOfMemoryError:
                        logging.error(f"❌ CUDA Out of Memory persistente en episodio {episode}. Deteniendo entrenamiento.")
                        # Guardar checkpoint con último estado disponible
                        trainer.save_checkpoint(
                            episode - 1 if episode > 0 else 0,
                            current_metrics=last_metrics,
                            current_loss=last_loss,
                            is_best=False
                        )
                        raise
                else:
                    raise
            except KeyboardInterrupt:
                logging.info("Entrenamiento interrumpido por el usuario.")
                # Guardar checkpoint con las últimas métricas disponibles
                trainer.save_checkpoint(
                    episode,
                    current_metrics=last_metrics,
                    current_loss=last_loss,
                    is_best=False
                )
                raise
            except Exception as e:
                logging.error(f"Error en episodio {episode}: {e}", exc_info=True)
                continue
        
        # Guardar checkpoint final (usar métricas del último episodio)
        final_survival = last_metrics.get('survival', 0.0)
        final_symmetry = last_metrics.get('symmetry', 0.0)
        final_combined = (10.0 * final_survival) + (5.0 * final_symmetry)
        is_final_best = final_combined < best_combined_metric
        
        trainer.save_checkpoint(
            total_episodes - 1,
            current_metrics=last_metrics,
            current_loss=last_loss,
            is_best=is_final_best
        )
        
        # Actualizar tiempo de entrenamiento en config
        training_duration = time.time() - training_start_time
        try:
            exp_config = load_experiment_config(exp_cfg.EXPERIMENT_NAME)
            if exp_config:
                config_dict = {}
                for key, value in vars(exp_config).items():
                    if isinstance(value, SimpleNamespace):
                        config_dict[key] = {k: v for k, v in vars(value).items()}
                    else:
                        config_dict[key] = value
                
                existing_total = config_dict.get('total_training_time', 0)
                config_dict['total_training_time'] = existing_total + training_duration
                config_dict['last_training_time'] = datetime.now().isoformat()
                config_dict['updated_at'] = datetime.now().isoformat()
                config_dict['TRAINER_VERSION'] = 'v4'  # Guardar versión usada
                
                save_experiment_config(exp_cfg.EXPERIMENT_NAME, config_dict, is_update=True)
                logging.info(f"Tiempo total de entrenamiento: {training_duration:.2f} segundos ({training_duration/3600:.2f} horas)")
        except Exception as e:
            logging.warning(f"No se pudo actualizar tiempo de entrenamiento: {e}")
        
        logging.info("Entrenamiento V4 completado.")
        print("✅ Entrenamiento completado.", flush=True)
        
    except KeyboardInterrupt:
        training_duration = time.time() - training_start_time
        logging.info("Entrenamiento interrumpido. Guardando checkpoint final...")
        # Guardar checkpoint con las últimas métricas disponibles
        trainer.save_checkpoint(
            episode,
            current_metrics=last_metrics,
            current_loss=last_loss,
            is_best=False
        )
        raise
