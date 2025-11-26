## 2024-11-21 - Manejo Robusto de CUDA Out of Memory

### Contexto
Durante el entrenamiento de modelos grandes (especialmente UNetConvLSTM), se reportó un error de `torch.cuda.OutOfMemoryError` que detenía completamente el entrenamiento, perdiendo todo el progreso. El error ocurría típicamente después de varios episodios cuando la memoria CUDA se fragmentaba o acumulaba.

### Problema Resuelto

#### Antes
- No había manejo de errores para OutOfMemoryError
- El entrenamiento se detenía abruptamente sin guardar progreso
- No había limpieza periódica de memoria CUDA
- La memoria se acumulaba durante episodios largos

#### Después
- ✅ Manejo robusto de OutOfMemoryError con reintento automático
- ✅ Limpieza periódica de caché CUDA durante entrenamiento
- ✅ Guardado automático de checkpoint si error persistente
- ✅ Recuperación automática después de limpiar memoria

### Implementación

#### 1. Manejo en `train_episode()` (QC_Trainer_v4)

**Archivo:** `src/trainers/qc_trainer_v4.py`

**Función:** `train_episode()`

**Cambios:**
- Envuelve `loss.backward()` y `optimizer.step()` en try-except para capturar OutOfMemoryError
- Si ocurre error, limpia caché CUDA y reintenta una vez
- Limpieza periódica de caché CUDA cada 10 episodios (después de calcular pérdida)

**Código:**
```python
try:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.motor.operator.parameters(), 1.0)
    self.optimizer.step()
except torch.cuda.OutOfMemoryError as e:
    # Limpiar caché y reintentar una vez
    logging.warning(f"⚠️ CUDA Out of Memory durante entrenamiento episodio {episode_num}. Limpiando caché...")
    torch.cuda.empty_cache()
    gc.collect()
    try:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.motor.operator.parameters(), 1.0)
        self.optimizer.step()
        logging.info("✅ Recuperado después de limpiar caché CUDA")
    except torch.cuda.OutOfMemoryError:
        logging.error(f"❌ CUDA Out of Memory persistente en episodio {episode_num}. Deteniendo entrenamiento.")
        raise

# Limpiar caché CUDA periódicamente (cada 10 episodios)
if episode_num % 10 == 0 and torch.cuda.is_available():
    torch.cuda.empty_cache()
```

#### 2. Manejo en Loop Principal de Entrenamiento

**Archivo:** `src/pipelines/pipeline_train.py`

**Función:** `_run_v4_training_loop()`

**Cambios:**
- Captura OutOfMemoryError en cada episodio del loop principal
- Limpia memoria y reintenta el episodio completo
- Guarda checkpoint antes de detener si error persistente
- Limpieza periódica cada 20 episodios o después de guardar checkpoint

**Código:**
```python
for episode in range(start_episode, total_episodes):
    try:
        loss, metrics = trainer.train_episode(episode)
        # ... logging y guardado ...
    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"❌ CUDA Out of Memory en episodio {episode}: {e}")
        # Limpiar y reintentar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            try:
                loss, metrics = trainer.train_episode(episode)
                logging.info(f"✅ Episodio {episode} completado después de limpiar memoria")
            except torch.cuda.OutOfMemoryError:
                # Guardar checkpoint y detener
                trainer.save_checkpoint(episode - 1 if episode > 0 else 0, ...)
                raise
    
    # Limpiar caché periódicamente
    if (episode + 1) % 20 == 0 or (episode + 1) % save_every == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
```

### Estrategias de Limpieza de Memoria

1. **Limpieza Periódica:**
   - Cada 10 episodios en `train_episode()` (después de calcular pérdida)
   - Cada 20 episodios en loop principal
   - Después de guardar cada checkpoint

2. **Limpieza Reactiva:**
   - Cuando ocurre OutOfMemoryError (antes de reintentar)
   - Después de eliminar `psi_history` (ya existía)

3. **Recuperación Automática:**
   - Reintento inmediato después de limpiar memoria
   - Si persiste, guarda checkpoint y detiene gracefulmente

### Beneficios

- ✅ **Reducción de errores:** Limpieza periódica previene acumulación de memoria
- ✅ **Recuperación automática:** Reintento después de limpiar memoria
- ✅ **Preservación de progreso:** Guarda checkpoint antes de detener si error persistente
- ✅ **Mejor estabilidad:** Menos interrupciones durante entrenamientos largos

### Consideraciones

- La limpieza periódica añade un pequeño overhead (~1-2ms por episodio)
- El reintento puede duplicar el tiempo de un episodio si ocurre error
- Si el error persiste después del reintento, indica que el modelo es demasiado grande para la GPU disponible

### Soluciones Alternativas si Persiste

Si el error persiste frecuentemente:
1. **Reducir tamaño del modelo:** `hid_dim`, `num_layers`, etc.
2. **Reducir tamaño del grid:** `GRID_SIZE_TRAINING` (ej: 64 → 32)
3. **Reducir pasos QCA:** `QCA_STEPS_TRAINING` (ej: 100 → 50)
4. **Usar mixed precision:** `torch.cuda.amp` (entrenamiento con FP16)
5. **Gradient checkpointing:** Ya comentado en código, se puede activar

### Referencias
- [[NATIVE_ENGINE_PERFORMANCE_ISSUES]]
- [[CHECKPOINT_STATE_ANALYSIS]]
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
