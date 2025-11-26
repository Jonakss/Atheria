## 2025-11-21 - Fix: Carga de Modelos en Servidor de Inferencia

### Problema
El servidor fallaba al cargar modelos desde el frontend con dos errores:
1. `AttributeError: module 'src.config' has no attribute 'D_STATE'`
2. `TypeError: load_model() got an unexpected keyword argument 'device'`

### Causa Raíz
- **Error 1**: El código usaba `global_cfg.D_STATE` que no existe. El atributo correcto es `MODEL_PARAMS['d_state']` desde la configuración del experimento.
- **Error 2**: La firma de `load_model()` cambió de `load_model(exp_name, device=device)` a `load_model(exp_cfg, checkpoint_path)`.

### Solución
**Archivo Modificado:** `src/pipelines/handlers/inference_handlers.py`

1. **Motor Nativo (C++)**:
   - Cargar configuración del experimento con `load_experiment_config(exp_name)`
   - Usar `exp_cfg.MODEL_PARAMS.d_state` en lugar de `global_cfg.D_STATE`
   - Llamar `load_model(exp_cfg, checkpoint_path)` con la firma correcta

2. **Motor Python**:
   - Cargar configuración del experimento
   - Crear modelo con `load_model(exp_cfg, checkpoint_path)`
   - Envolver en `Aetheria_Motor` con parámetros correctos

### Resultado
- ✅ Carga de modelos funciona correctamente
- ✅ Compatibilidad con motor nativo y Python
- ✅ Configuración del experimento se carga dinámicamente

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
