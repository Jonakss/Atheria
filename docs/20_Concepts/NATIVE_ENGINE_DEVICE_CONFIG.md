# Configuración del Device del Motor Nativo

## Descripción

El motor nativo C++ puede ejecutarse en CPU o GPU (CUDA). Por defecto, detecta automáticamente el mejor dispositivo disponible, pero puede ser forzado mediante configuración.

## Parámetros

### Variable de Entorno

- **`ATHERIA_NATIVE_DEVICE`**: Variable de entorno para forzar el device
  - Valores posibles: `auto`, `cpu`, `cuda`
  - Por defecto: `auto`

### Configuración en Python

```python
from src import config

# Auto-detección (por defecto)
device = config.get_native_device()  # 'cpu' o 'cuda' según disponibilidad

# Forzar CPU
config.NATIVE_ENGINE_DEVICE = 'cpu'
device = config.get_native_device()  # Siempre 'cpu'

# Forzar CUDA
config.NATIVE_ENGINE_DEVICE = 'cuda'
device = config.get_native_device()  # 'cuda' o 'cpu' si no está disponible
```

## Auto-detección

Cuando `NATIVE_ENGINE_DEVICE` es `"auto"` (por defecto):

1. Intenta detectar CUDA disponible
2. Verifica que realmente funciona (`torch.cuda.device_count() > 0`)
3. Si CUDA está disponible y funcional, usa `"cuda"`
4. Si no, usa `"cpu"` como fallback

## Forzar Device

### Forzar CPU

```bash
export ATHERIA_NATIVE_DEVICE=cpu
python3 scripts/test_native_engine.py --experiment ...
```

### Forzar CUDA

```bash
export ATHERIA_NATIVE_DEVICE=cuda
python3 scripts/test_native_engine.py --experiment ...
```

### Auto-detección (por defecto)

```bash
# No es necesario establecer la variable
export ATHERIA_NATIVE_DEVICE=auto  # Opcional
python3 scripts/test_native_engine.py --experiment ...
```

## Uso en Código

```python
from src.engines.native_engine_wrapper import NativeEngineWrapper

# Auto-detección (usa config.get_native_device())
wrapper = NativeEngineWrapper(
    grid_size=64,
    d_state=8,
    device=None,  # None = auto-detección
    cfg=exp_config
)

# Forzar CPU
wrapper = NativeEngineWrapper(
    grid_size=64,
    d_state=8,
    device='cpu',  # Forzar CPU
    cfg=exp_config
)

# Forzar CUDA
wrapper = NativeEngineWrapper(
    grid_size=64,
    d_state=8,
    device='cuda',  # Forzar CUDA
    cfg=exp_config
)
```

## Notas

- Si se fuerza CUDA pero no está disponible, automáticamente usa CPU como fallback
- El device se cachea en la primera llamada a `get_native_device()` para evitar múltiples verificaciones
- La auto-detección es la opción recomendada para la mayoría de casos de uso

## Referencias

- `src/config.py::get_native_engine_device()`
- `src/config.py::get_native_device()`
- `src/engines/native_engine_wrapper.py::__init__()`

