## 2025-11-27 - Tool: Progressive Training Notebook (Long-Running GPU Sessions)

### Contexto
Creación de notebook Jupyter optimizado para entrenamientos largos (6-24 horas) en Google Colab y Kaggle, con aprovechamiento máximo de cuota de GPU.

### Motivación
- **Limitaciones de notebooks existentes**: El notebook `Atheria_Training_Kaggle_Colab.ipynb` no estaba optimizado para sesiones largas
- **Cuota de GPU**: Colab Free (~12h/día), Colab Pro (~24h), Kaggle (30h/semana) no se aprovechaban al máximo
- **Falta de persistencia**: Si la sesión se desconectaba, se perdía todo el progreso
- **Sin monitoreo**: No había forma de saber si estaba cerca del límite de tiempo
- **Checkpointing manual**: Usuario tenía que guardar manualmente en Drive

### Archivo Creado

**`notebooks/Atheria_Progressive_Training.ipynb`**

Notebook completo con 9 secciones principales:

#### Características Implementadas

**1. Auto-guardado en Google Drive** 🔄
- Montaje automático de Drive (solo Colab)
- Sincronización configurable cada N episodios
- Estructura de carpetas organizada:
  - `/MyDrive/Atheria/checkpoints/{experiment}/`
  - `/MyDrive/Atheria/logs/{experiment}/`
  - `/MyDrive/Atheria/exports/`

**2. Monitoreo de Recursos en Tiempo Real** 📊
- Clase `ResourceMonitor` personalizada
- Métricas cada 10 episodios:
  - GPU Utilization (%)
  - GPU Memory (GB usado/reservado)
  - RAM usage (GB/%)
  - Tiempo transcurrido vs restante
- Alerta automática al 90% del límite de tiempo

**3. Auto-Recuperación Inteligente** ⚡
- Detección automática de checkpoints en Drive
- Variable `AUTO_RESUME=True` para continuar desde último checkpoint
- Extracción de episodio desde checkpoint para calcular progreso restante
- Transparente para el usuario (solo ejecutar de nuevo)

**4. Límite de Tiempo Automático** ⏰
- Configuración de `MAX_TRAINING_HOURS` (default: 10h)
- Verificación ANTES de cada episodio si hay tiempo suficiente
- Guardado de emergencia automático al 90% del límite
- Evita timeouts de Colab/Kaggle

**5. Smart Checkpointing** 💾
- Usa `QC_Trainer_v4` con sistema de retención inteligente
- Solo guarda mejores N modelos (default: 5)
- Siempre guarda `last_model.pth` para continuidad
- Sincronización a Drive cada M episodios (default: 50)
- Checkpoints locales cada N episodios (default: 10)

**6. Visualización en Tiempo Real** 📈
- Gráfico de pérdida actualizado cada 10 episodios
- Historial completo de entrenamiento
- 4 gráficos finales:
  - Evolución de pérdida
  - Tasa de supervivencia
  - Timeline de checkpoints
  - Distribución de pérdidas

**7. Exportación Completa** 📦
- Modelo final a TorchScript (`.pt`)
- Reporte de entrenamiento (Markdown)
- Logs JSON con toda la información
- Gráficos PNG guardados en Drive

### Decisiones de Diseño

#### ¿Por qué clase ResourceMonitor separada?
- **Portabilidad**: Funciona en Colab, Kaggle y local
- **Reutilizable**: Fácil de adaptar para otros notebooks
- **Testeable**: Puede verificarse independientemente
- **Modular**: No acopla lógica de monitoreo con entrenamiento

#### ¿Por qué sincronización en dos niveles (local + Drive)?
- **Performance**: Guardar en local es instantáneo (~50ms)
- **Seguridad**: Guardar en Drive protege contra desconexiones
- **Balance**: Checkpoints locales frecuentes, sync a Drive menos frecuente
- **Trade-off**: Local rápido pero volátil vs Drive lento pero persistente

#### ¿Por qué verificar tiempo ANTES de cada episodio?
- **Prevención**: Evita empezar un episodio que no terminará
- **Guardado limpio**: Garantiza tiempo suficiente para guardar checkpoint
- **Sin pérdida**: Usuario no pierde progreso si se acerca al límite
- **Margen de seguridad**: Detiene al 90% del límite (10% buffer)

### Configuración Recomendada

**Colab Free (12h/día):**
```python
"MAX_TRAINING_HOURS": 10,
"TOTAL_EPISODES": 500-800,
"SAVE_EVERY_EPISODES": 10,
"DRIVE_SYNC_EVERY": 50,
```

**Colab Pro (24h continuas):**
```python
"MAX_TRAINING_HOURS": 20,
"TOTAL_EPISODES": 1500-2000,
"SAVE_EVERY_EPISODES": 20,
"DRIVE_SYNC_EVERY": 100,
```

**Kaggle (30h/semana):**
```python
"MAX_TRAINING_HOURS": 9,  # Por sesión (3 sesiones/semana)
"TOTAL_EPISODES": 800-1000,
"SAVE_EVERY_EPISODES": 10,
"DRIVE_SYNC_EVERY": 0,  # No hay Drive, usar /kaggle/working
```

### Workflow de Usuario

**Primera sesión:**
1. Configurar `EXPERIMENT_NAME` y parámetros
2. Ejecutar todas las celdas (Runtime → Run all)
3. Dejar corriendo sin supervisión
4. Notebook se auto-detiene antes de timeout

**Sesiones posteriores:**
1. Mantener `AUTO_RESUME=True`
2. Ejecutar todas las celdas de nuevo
3. Continúa automáticamente desde episodio guardado

### Documentación Creada

**`docs/99_Templates/PROGRESSIVE_TRAINING_GUIDE.md`**

Guía completa de usuario con:
- Preparación de Google Drive
- Estrategias de configuración (rápido/estándar/largo)
- Troubleshooting detallado:
  - Drive sync lento
  - RAM insuficiente
  - GPU subutilizada
  - Checkpoints no encontrados
  - Timeouts
- Mejores prácticas:
  - Sesiones múltiples en Colab Free
  - Monitoreo externo opcional
  - Validación periódica
  - Estimaciones de tiempo
- 3 ejemplos completos paso a paso

### Ventajas vs Notebook Anterior

| Característica | Notebook Anterior | Notebook Progresivo |
|----------------|-------------------|---------------------|
| Drive Integration | ❌ Manual | ✅ Automático |
| Resource Monitoring | ❌ No | ✅ GPU/RAM/Tiempo |
| Auto-Resume | ❌ No | ✅ Detección automática |
| Time Limits | ❌ No | ✅ Guardado de emergencia |
| Visualization | ✅ Básica | ✅ Tiempo real + final |
| Checkpoint Strategy | ✅ Smart Save | ✅ Smart Save + Drive sync |
| Uso típico | 1-3 horas | 6-24 horas |

### Limitaciones Conocidas

⚠️ **Kaggle**: No tiene Drive nativo, usa almacenamiento local (`/kaggle/working/`)
- Usuario debe descargar checkpoints manualmente al finalizar
- Alternativa: Usar Kaggle Datasets API para persistencia

⚠️ **Colab Free**: Límite variable (~12h/día)
- Puede cambiar según carga de Google
- Recomendar usar en horarios de baja demanda

⚠️ **Notebook no testeable automáticamente**
- Requiere entorno interactivo Colab/Kaggle con GPU
- Testing manual requerido por usuario

### Archivos Creados/Modificados

**Notebooks:**
- `notebooks/Atheria_Progressive_Training.ipynb` - Notebook principal (NUEVO)

**Documentación:**
- `docs/99_Templates/PROGRESSIVE_TRAINING_GUIDE.md` - Guía completa de usuario (NUEVO)
- `docs/40_Experiments/AI_DEV_LOG.md` - Esta entrada

**Artifacts:**
- `.gemini/antigravity/brain/.../implementation_plan.md` - Plan de implementación
- `.gemini/antigravity/brain/.../task.md` - Checklist de tareas

### Extensiones Futuras

- [ ] Integración con Weights & Biases para tracking externo
- [ ] Notificaciones por email al completar (Colab API)
- [ ] Kaggle Datasets API para persistencia automática
- [ ] Compresión automática de checkpoints antiguos en Drive
- [ ] Dashboard web externo para monitoreo remoto
- [ ] Auto-ajuste de `DRIVE_SYNC_EVERY` basado en velocidad de sync

### Referencias

- [[PROGRESSIVE_TRAINING_GUIDE]] - Guía de usuario completa
- [[QC_TRAINER_V4]] - Trainer con Smart Save usado
- `notebooks/Atheria_Training_Kaggle_Colab.ipynb` - Notebook base anterior
- `src/trainers/qc_trainer_v4.py` - Lógica de entrenamiento
- `src/model_loader.py` - Carga y exportación de modelos
