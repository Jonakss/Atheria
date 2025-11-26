# Compatibilidad: V3/V4 con Motor Nativo

**Fecha**: 2024-11-19  
**Objetivo**: Verificar que los modelos entrenados con QC_Trainer_v3 y QC_Trainer_v4 sean compatibles con el motor nativo de alto rendimiento.

## Respuesta: ✅ SÍ, AMBOS SON COMPATIBLES

### ¿Por qué son compatibles?

1. **Mismo formato de checkpoint**: Tanto V3 como V4 guardan modelos en formato `.pth` con la misma estructura:
   ```python
   {
       'model_state_dict': {...},  # Pesos del modelo
       'episode': ...,             # Episodio actual
       'optimizer_state_dict': {...},  # Estado del optimizador
       ...
   }
   ```

2. **Misma arquitectura de modelos**: Ambos trainers usan las mismas clases de modelos (UNet, UNetUnitary, DeepQCA, etc.) definidas en `src/models/`.

3. **Exportación a TorchScript**: El script `export_model_to_jit.py` puede exportar cualquier checkpoint (v3 o v4) a TorchScript (`.pt`), independientemente del trainer que lo entrenó.

4. **Motor nativo agnóstico**: El motor nativo `Engine` en C++ solo necesita:
   - Un modelo exportado en formato TorchScript (`.pt`)
   - La configuración correcta (d_state, hidden_channels, model_type)
   - No importa si fue entrenado con v3 o v4

### Workflow para usar v3/v4 con motor nativo:

```
1. Entrenar modelo con v3 o v4
   ↓
   output/training_checkpoints/<exp_name>/checkpoint_ep*.pth

2. Exportar a TorchScript
   ↓
   python scripts/export_model_to_jit.py checkpoint.pth --experiment_name <exp_name>
   ↓
   output/training_checkpoints/<exp_name>/model_jit.pt

3. Usar en motor nativo
   ↓
   engine = atheria_core.Engine(d_state=4, device="cpu")
   engine.load_model("model_jit.pt")
   engine.step_native()  # 250-400x más rápido!
```

### Ejemplo práctico:

```python
import atheria_core
import torch
from src.utils import get_latest_checkpoint, get_latest_jit_model

# 1. Buscar checkpoint (v3 o v4, no importa)
experiment_name = "MiExperimento"
checkpoint_path = get_latest_checkpoint(experiment_name)

# 2. Exportar a JIT (solo una vez)
if not get_latest_jit_model(experiment_name, silent=True):
    from scripts.export_model_to_jit import export_model_to_jit
    from src.utils import load_experiment_config
    
    exp_config = load_experiment_config(experiment_name)
    d_state = exp_config.MODEL_PARAMS.d_state
    hidden_channels = exp_config.MODEL_PARAMS.hidden_channels
    model_type = exp_config.MODEL_ARCHITECTURE
    
    export_model_to_jit(
        checkpoint_path,
        experiment_name=experiment_name,
        d_state=d_state,
        hidden_channels=hidden_channels,
        model_type=model_type
    )

# 3. Usar en motor nativo
jit_path = get_latest_jit_model(experiment_name)
engine = atheria_core.Engine(d_state, "cpu")
engine.load_model(jit_path)

# ¡Ahora tienes 250-400x más rendimiento!
for _ in range(1000):
    count = engine.step_native()
```

### Diferencias entre v3 y v4 (no afectan compatibilidad):

| Aspecto | V3 | V4 |
|---------|----|----|
| Función de pérdida | Recompensa clásica | Multi-objetivo (survival, symmetry, complexity) |
| Ruido durante entrenamiento | No | Sí (QuantumNoiseInjector) |
| Curriculum Learning | No | Sí (ruido aumenta con episodios) |
| Formato checkpoint | `.pth` | `.pth` (mismo formato) |
| Compatible con motor nativo | ✅ Sí | ✅ Sí |

### Notas importantes:

1. **Versión del trainer NO se guarda en el checkpoint**: Solo se guardan los pesos del modelo, no quién lo entrenó.

2. **El motor nativo no necesita saber si es v3 o v4**: Solo necesita el modelo en formato TorchScript.

3. **Performance es la misma**: Un modelo entrenado con v3 tendrá el mismo rendimiento en el motor nativo que uno entrenado con v4 (ambos 250-400x más rápido que Python).

4. **Mejores prácticas**:
   - Exporta a JIT después de entrenar (una vez)
   - Usa `get_latest_jit_model()` para buscar modelos automáticamente
   - Si no existe modelo JIT, el script puede exportarlo automáticamente

### Conclusión:

**✅ Sí, puedes usar modelos v3 en el motor nativo de alto rendimiento.**

El motor nativo es completamente agnóstico respecto al trainer usado. Lo único que importa es:
- Que el modelo esté en formato TorchScript (`.pt`)
- Que la configuración (d_state, etc.) sea correcta

**Todos los modelos entrenados con v3 o v4 funcionan perfectamente con el motor nativo.**

