# ... existing code ...

def get_latest_jit_model(experiment_name: str, silent: bool = False) -> str | None:
    """
    Encuentra la ruta al último modelo JIT exportado (.pt) para un experimento dado.
    
    Los modelos JIT se guardan en: output/training_checkpoints/<experiment_name>/model_*.pt
    o en: output/jit_models/<experiment_name>/model.pt
    
    Args:
        experiment_name: Nombre del experimento
        silent: Si es True, no loguea warnings cuando no hay modelos
    """
    # Buscar en directorio de checkpoints (mismo lugar que checkpoints)
    checkpoint_dir = os.path.join(global_cfg.TRAINING_CHECKPOINTS_DIR, experiment_name)
    
    # Buscar modelos JIT
    jit_models = []
    
    # Opción 1: En el directorio de checkpoints
    if os.path.exists(checkpoint_dir):
        jit_models.extend([
            os.path.join(checkpoint_dir, f) 
            for f in os.listdir(checkpoint_dir) 
            if f.endswith('.pt') and (f.startswith('model_') or f == 'model.pt')
        ])
    
    # Opción 2: En directorio dedicado de modelos JIT
    jit_dir = os.path.join(global_cfg.OUTPUT_DIR, "jit_models", experiment_name)
    if os.path.exists(jit_dir):
        jit_models.extend([
            os.path.join(jit_dir, f) 
            for f in os.listdir(jit_dir) 
            if f.endswith('.pt')
        ])
    
    if not jit_models:
        if not silent:
            logging.warning(f"No se encontraron modelos JIT (.pt) para '{experiment_name}'")
            logging.info(f"Busca modelos en: {checkpoint_dir} o {jit_dir}")
        return None
    
    # Retornar el más reciente (por timestamp)
    latest_model = max(jit_models, key=lambda p: os.path.getmtime(p))
    if not silent:
        logging.info(f"Modelo JIT más reciente encontrado: {latest_model}")
    return latest_model

# ... existing code ...
