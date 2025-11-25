# src/pipeline_train.py
import torch
from types import SimpleNamespace
import logging
import os

from .. import config as global_cfg
from ..model_loader import create_new_model, load_model
from ..engines.qca_engine import Aetheria_Motor
from ..trainers import QC_Trainer_v3, QC_Trainer_v4
from ..utils import get_latest_checkpoint, load_experiment_config

def run_training_pipeline(exp_cfg: SimpleNamespace, checkpoint_path: str | None = None):
    """
    Ejecuta el pipeline de entrenamiento completo para una configuración de experimento dada.
    
    Args:
        exp_cfg: Configuración del experimento (SimpleNamespace)
        checkpoint_path: Ruta al checkpoint si se está continuando entrenamiento (opcional)
    """
    device = global_cfg.get_device()
    logging.info(f"Pipeline de entrenamiento iniciado en el dispositivo: {device}")

    # Determinar versión del trainer antes de cargar el modelo
    # (V4 puede crear su propio modelo desde cero, V3 siempre necesita que se cargue primero)
    trainer_version = _determine_trainer_version(exp_cfg)
    logging.info(f"Usando entrenador: {trainer_version}")

    # 1. Cargar o crear el modelo
    # Para V3: siempre cargar/crear el modelo aquí
    # Para V4: solo cargar si hay checkpoint o transfer learning, sino V4 lo crea internamente
    ley_M = None
    if checkpoint_path:
        # Continuar entrenamiento: cargar modelo con checkpoint del mismo experimento
        ley_M, state_dict = load_model(exp_cfg, checkpoint_path)
        if ley_M is None:
            logging.error("No se pudo cargar el modelo desde el checkpoint. Abortando entrenamiento.")
            return
        # El modelo ya está en modo eval después de load_model, cambiar a train
        ley_M.train()
        logging.info(f"Modelo cargado desde checkpoint: {checkpoint_path}")
    elif hasattr(exp_cfg, 'LOAD_FROM_EXPERIMENT') and exp_cfg.LOAD_FROM_EXPERIMENT:
        # Transfer Learning: crear nuevo modelo y cargar pesos de otro experimento
        try:
            ley_M = create_new_model(exp_cfg)
            logging.info(f"Nuevo modelo creado. Cargando pesos desde '{exp_cfg.LOAD_FROM_EXPERIMENT}' para transfer learning...")
            
            # Cargar checkpoint del experimento base
            from ..utils import get_latest_checkpoint
            base_checkpoint = get_latest_checkpoint(exp_cfg.LOAD_FROM_EXPERIMENT)
            if base_checkpoint:
                try:
                    checkpoint = torch.load(base_checkpoint, map_location=device)
                    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
                    
                    # Manejar prefijo _orig_mod. si existe
                    if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
                        cleaned_state_dict = {}
                        for key, value in model_state_dict.items():
                            if key.startswith('_orig_mod.'):
                                cleaned_state_dict[key.replace('_orig_mod.', '', 1)] = value
                            else:
                                cleaned_state_dict[key] = value
                        model_state_dict = cleaned_state_dict
                    
                    # Cargar con strict=False para permitir diferencias de arquitectura
                    ley_M.load_state_dict(model_state_dict, strict=False)
                    logging.info(f"✅ Transfer learning: pesos cargados desde '{exp_cfg.LOAD_FROM_EXPERIMENT}'. Iniciando desde episodio 0.")
                except Exception as e:
                    logging.warning(f"⚠️ No se pudieron cargar pesos desde '{exp_cfg.LOAD_FROM_EXPERIMENT}': {e}. Iniciando desde cero.")
            else:
                logging.warning(f"⚠️ No se encontró checkpoint para '{exp_cfg.LOAD_FROM_EXPERIMENT}'. Iniciando desde cero.")
            
            ley_M.train()
        except Exception as e:
            logging.error(f"Error al crear el modelo: {e}", exc_info=True)
            return
    elif trainer_version == "v3":
        # Para V3, siempre necesitamos crear el modelo (aunque sea desde cero)
        # V4 puede crear su propio modelo, así que solo lo creamos aquí para V3
        try:
            ley_M = create_new_model(exp_cfg)
            logging.info("Nuevo modelo creado para entrenamiento desde cero (V3).")
            ley_M.train()
        except Exception as e:
            logging.error(f"Error al crear el modelo: {e}", exc_info=True)
            return
    # Para V4 desde cero, no creamos el modelo aquí - V4 lo crea internamente

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
