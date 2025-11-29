# Configuración Low-Memory para GPUs Pequeñas

Este directorio contiene configuraciones optimizadas para entrenar en GPUs con memoria limitada (<4 GiB).

## Características

### Optimizaciones Implementadas

1. **Mixed Precision Training (FP16)**
   - Reduce uso de memoria ~50%
   - Implementado automáticamente con PyTorch AMP
   - Se activa cuando `use_amp=True` en GPU

2. **CUDA Memory Allocator**  
   - Variable de entorno: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - Reduce fragmentación de memoria
   - Configurado automáticamente al iniciar entrenamiento

3. **Arquitectura Reducida**
   - `d_state`: 10 (vs 11-14)
   - `hidden_channels`: 32 (vs 64-128)
   - `qca_steps`: 200 (vs 500)

4. **Cache Management Agresivo**
   - Vaciar caché CUDA cada 5 episodios (vs 10)
   - Garbage collection explícito después de cada episodio

### GPU Memory Detection

El trainer detecta automáticamente si tu GPU tiene <4 GiB y muestra advertencias con recomendaciones.

## Uso

### Desde CLI

```bash
python -m src.trainer \
  --experiment_name "LOW_MEM_TEST" \
  --model_architecture "MLP" \
  --model_params '{"d_state": 10, "hidden_channels": 32, "activation": "SiLU"}' \
  --lr_rate_m 0.0001 \
  --grid_size_training 32 \
  --qca_steps_training 200 \
  --total_episodes 5000 \
  --noise_level 0.05
```

### Desde Experiment Config

```bash
python scripts/run_experiment.py --config experiments/low_memory_mlp.yaml
```

## Resultados Esperados

### Uso de Memoria (GPU 3.68 GiB)

| Configuración | Memoria Usada | Estado |
|--------------|---------------|--------|
| **Default** (d=14, h=128, steps=500) | ~3.56 GiB | ❌ OOM |
| **Low-Memory** (d=10, h=32, steps=200) | ~1.8 GiB | ✅ OK |
| **Low-Memory + FP16** | ~0.9 GiB | ✅✅ Óptimo |

### Performance

- **FP16**: Velocidad similar o ligeramente mayor que FP32
- **Arquitectura reducida**: ~3x más rápido por episodio
- **Calidad**: Se espera convergencia similar para problemas simples

## Troubleshooting

### Si aún obtienes OOM:

1. **Reduce `qca_steps`** a 100
2. **Reduce `grid_size`** a 16
3. **Reduce `d_state`** a 8
4. **Habilita Gradient Checkpointing** (trade compute for memory):
   ```python
   USE_GRADIENT_CHECKPOINTING = True
   ```

### Verificar uso de memoria:

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

## Referencias

- [[CUDA_MEMORY_OPTIMIZATION]] - Documentación completa
- [[QC_Trainer_v4]] - Implementación del trainer
- [[config_low_memory.py]] - Parámetros de configuración
