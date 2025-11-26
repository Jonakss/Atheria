# Experimento: SNN_UNET-d32-h128-g16-lr1e-4

**Fecha de Creación:** 2025-11-24 23:12:11

## Configuración del Experiment

```json
{
  "experiment_name": "SNN_UNET-d32-h128-g16-lr1e-4",
  "model_architecture": "UNKNOWN",
  "lr": 0.0001,
  "grid_size": 16,
  "qca_steps": 250,
  "gamma_decay": 0.01,
  "d_state": 32,
  "max_checkpoints_to_keep": 5,
  "max_noise": 0.05
}
```

## Historial de Resultados

### Tabla de Hitos (Mejores Checkpoints)

| Episodio | Fecha | Loss Total | Survival | Symmetry | Complexity | Métrica Combinada | Checkpoint |
|----------|-------|------------|----------|----------|------------|-------------------|------------|
| 49 | 2025-11-24 23:12:55 | 4.099892 | 0.200000 | 0.012612 | 2.036832 | 2.063060 | `output/training_checkpoints/SNN_UNET-d32-h128-g16-lr1e-4/best_model.pth` |
