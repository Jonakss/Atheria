# Optimización de Memoria CUDA para GPUs Pequeñas

**Estado:** Backlog (Pendiente)  
**Prioridad:** Media  
**Complejidad:** Alta  
**Estimación:** 2-3 días

## Contexto

Durante el entrenamiento en GPU de 3.68 GiB se produce error `CUDA Out of Memory`:
- Modelo actual: UNetUnitary con 27.8M parámetros
- Configuración: grid_size=64, d_state=14, hidden_channels=128, qca_steps=500
- Memoria usada: 3.56 GiB / 3.68 GiB (quedando solo 11.75 MiB libres)

## Problema Técnico

```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 3.68 GiB 
of which 11.75 MiB is free.
```

El error ocurre en el forward pass del UNet durante `train_episode()`.

## Soluciones Propuestas

### 1. Mixed Precision Training (FP16)
**Impacto:** Reduce memoria ~50%  
**Trade-off:** Posible pérdida de precisión numérica

Implementar Automatic Mixed Precision (AMP) con `torch.cuda.amp`:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(input)
scaler.scale(loss).backward()
```

### 2. Gradient Checkpointing
**Impacto:** Reduce memoria ~30-40%  
**Trade-off:** Aumenta tiempo de entrenamiento ~30%

Envolver bloques del UNet con `torch.utils.checkpoint`:
```python
if self.use_checkpointing and self.training:
    x1 = torch.utils.checkpoint.checkpoint(self.inc, x_cat, use_reentrant=False)
```

### 3. Reducir Arquitectura del Modelo
**Impacto:** Reduce memoria ~75%  
**Trade-off:** Reduce capacidad del modelo

Cambiar `hidden_channels` de 128 → 64:
- Parámetros: 27.8M → ~7M
- Memoria estimada: ~1.5 GiB

### 4. Variables de Entorno CUDA
**Impacto:** Reduce fragmentación  
**Trade-off:** Ninguno

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Archivos a Modificar

### Core
- `src/models/unet_unitary.py` - Agregar gradient checkpointing opcional
- `src/trainers/qc_trainer_v4.py` - Implementar mixed precision (AMP)
- `src/trainer.py` - Configurar variables de entorno y auto-detección de GPU

### Configuración
- `src/config_low_memory.py` (nuevo) - Configuración para GPUs <4 GiB
- `configs/experiments/experiment_low_memory.yaml` (nuevo) - Template optimizado

### Monitoring
- Agregar logging de uso de memoria CUDA en `train_episode()`
- Dashboard opcional para visualizar memoria en tiempo real

## Configuración Recomendada (Low Memory Profile)

```yaml
MODEL_PARAMS:
  d_state: 10           # Reducido de 14
  hidden_channels: 64   # Reducido de 128
  use_checkpointing: true

TRAINING:
  grid_size: 64
  qca_steps: 200        # Reducido de 500
  use_mixed_precision: true
  gradient_accumulation_steps: 2

CUDA_ALLOC_CONF: "expandable_segments:True"
```

**Uso de memoria estimado:** ~1.5 GiB (reducción de 58%)

## Plan de Verificación

### Criterios de Éxito
- ✅ Entrenamiento completa 100 episodios sin OOM
- ✅ Uso de memoria < 3.5 GiB constantemente
- ✅ Pérdida converge correctamente
- ✅ Métricas de calidad aceptables (survival < 1.0, symmetry < 0.5)

### Tests
1. Test unitario para mixed precision (comparar FP32 vs FP16)
2. Test de gradient checkpointing (validar gradientes)
3. Test de convergencia con configuración low_memory
4. Monitoreo con `nvidia-smi` durante entrenamiento

## Referencias

- [[UNET_UNITARY]] - Arquitectura del modelo
- [[QC_TRAINER_V4]] - Trainer actual
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- Gradient Checkpointing: https://pytorch.org/docs/stable/checkpoint.html

## Notas

- La solución **más rápida** es solo FP16 + env vars (mantiene arquitectura)
- La solución **más confiable** es reducir hidden_channels=64 + FP16
- Gradient checkpointing es opcional (si aún falla con FP16)

---

**Última actualización:** 2025-11-28  
**Relacionado con:** Entrenamiento en GPUs pequeñas, Memory optimization
