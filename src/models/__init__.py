import importlib
import pkgutil
import logging

_models = {}

def register_model(name):
    """A decorator to register a new model architecture."""
    def decorator(cls):
        logging.info(f"Registering model: {name}")
        _models[name] = cls
        return cls
    return decorator

def get_model_class(name):
    """Returns the model class for a given name."""
    model_class = _models.get(name)
    if model_class is None:
        raise ValueError(f"Model '{name}' not found. Available models: {list_models()}")
    return model_class

def list_models():
    """Returns a list of all registered model names."""
    return list(_models.keys())

# --- Auto-discovery of models ---
# Automatically import all modules in this package to trigger registration.
logging.info("Searching for models in 'src/models'...")
for _, module_name, _ in pkgutil.iter_modules(__path__):
    try:
        importlib.import_module(f".{module_name}", __name__)
    except Exception as e:
        logging.error(f"Failed to import model module '{module_name}': {e}")

logging.info(f"Found models: {list_models()}")
