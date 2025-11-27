# Guía: Entrenamiento Progresivo en Atheria 4

Esta guía explica cómo usar el notebook `Atheria_Progressive_Training.ipynb` para entrenamientos largos en Google Colab o Kaggle, optimizando el uso de cuota de GPU.

---

## 🎯 Casos de Uso

Este notebook es ideal para:
- Entrenamientos de **muchas horas** (6-24h)
- Aprovechar **cuota de GPU completa** (Colab/Kaggle)
- Experimentación sin supervisión
- Entrenamientos que necesitan **auto-recuperación** si se desconecta

---

## 📋 Preparación

### 1. Google Drive (solo Colab)

**Primera vez:**
1. El notebook montará automáticamente tu Drive
2. Creará estructura de carpetas en `/MyDrive/Atheria/`
3. Checkpoints se guardarán en `/MyDrive/Atheria/checkpoints/{experiment_name}/`

**Importante:**
- Los checkpoints en Drive permiten continuar si la sesión muere
- Asegúrate de tener **suficiente espacio** (~1-5GB por experimento)
- Drive sync puede tardar unos segundos, es normal

### 2. Entorno (Colab o Kaggle)

#### Google Colab
- **Free**: ~12 horas/día (variable según uso)
- **Pro**: ~24 horas continuas
- **GPU recomendada**: T4 (15GB VRAM)

#### Kaggle
- **Cuota**: 30 horas/semana
- **GPU disponibles**: T4, P100
- **Nota**: Checkpoints se guardan localmente, descargar al finalizar

---

## ⚙️ Configuración del Experimento

### Parámetros Clave (Sección 5 del notebook)

```python
EXPERIMENT_CONFIG = {
    # Identificación
    "EXPERIMENT_NAME": "UNET_64ch_D8_Progressive",
    
    # Arquitectura
    "MODEL_ARCHITECTURE": "UNET",  # UNET, SNN_UNET, MLP, DEEP_QCA
    "MODEL_PARAMS": {
        "d_state": 8,           # Dimensión estado cuántico (4, 8, 16, 32)
        "hidden_channels": 64,  # Canales ocultos (16, 32, 64, 128)
    },
    
    # Entrenamiento
    "GRID_SIZE_TRAINING": 64,     # 32, 64, 128
    "QCA_STEPS_TRAINING": 100,
    "LR_RATE_M": 1e-4,
    
    # Progresivo
    "TOTAL_EPISODES": 1000,       # Total a entrenar
    "SAVE_EVERY_EPISODES": 10,    # Checkpoint local cada N
    "DRIVE_SYNC_EVERY": 50,       # Sync a Drive cada N
    "MAX_TRAINING_HOURS": 10,     # Límite de tiempo
    
    # Auto-recuperación
    "AUTO_RESUME": True,          # Continuar desde Drive
}
```

### Estrategias de Configuración

#### 🐇 Rápido (pruebas, 1-2 horas)
```python
"TOTAL_EPISODES": 100,
"GRID_SIZE_TRAINING": 32,
"SAVE_EVERY_EPISODES": 5,
"DRIVE_SYNC_EVERY": 20,
"MAX_TRAINING_HOURS": 2,
```

#### 🐢 Estándar (entrenamiento normal, 6-8 horas)
```python
"TOTAL_EPISODES": 500,
"GRID_SIZE_TRAINING": 64,
"SAVE_EVERY_EPISODES": 10,
"DRIVE_SYNC_EVERY": 50,
"MAX_TRAINING_HOURS": 8,
```

#### 🐌 Largo (máxima calidad, 12-24 horas)
```python
"TOTAL_EPISODES": 2000,
"GRID_SIZE_TRAINING": 128,
"SAVE_EVERY_EPISODES": 20,
"DRIVE_SYNC_EVERY": 100,
"MAX_TRAINING_HOURS": 20,
```

---

## 🚀 Workflow Recomendado

### Primera Sesión (Desde Cero)

1. **Configurar experimento** (Sección 5)
   - Definir nombre único
   - Elegir arquitectura y parámetros
   - Configurar `AUTO_RESUME = True`

2. **Ejecutar todas las celdas**
   - Runtime → Run all (Colab)
   - Cell → Run All (Kaggle)

3. **Dejar corriendo sin supervisión**
   - El notebook se auto-guarda en Drive
   - Monitorea recursos automáticamente
   - Se detiene antes de timeout

4. **Verificar progreso** (opcional)
   - Cada 10 episodios: gráfico de pérdida
   - Cada 50 episodios: sync a Drive confirmado
   - Monitor de recursos actualizado

### Sesiones Posteriores (Continuar)

1. **Verificar `AUTO_RESUME = True`**
   - El notebook detecta automáticamente último checkpoint en Drive

2. **Ajustar `TOTAL_EPISODES` si necesario**
   - Ejemplo: Si ya completó 500, aumentar a 1000

3. **Ejecutar todas las celdas de nuevo**
   - Continúa automáticamente desde episodio guardado
   - No reinicia desde cero

4. **Repetir hasta convergencia**

---

## 📊 Monitoreo de Recursos

El notebook muestra automáticamente cada 10 episodios:

```
📊 RECURSOS:
  GPU Utilization: 85.3%
  GPU Memory: 8.42GB / 15.00GB
  RAM: 12.51GB / 25.50GB (49.1%)
  
⏰ TIEMPO:
  Transcurrido: 2:34:18
  Restante: 7:25:42 (de 10h máximo)
```

### Interpretación

- **GPU Utilization 80-100%**: ✅ Óptimo, GPU bien aprovechada
- **GPU Utilization 50-80%**: ⚠️ Puede mejorar, revisar grid size
- **GPU Utilization <50%**: ❌ Subutilización, aumentar complejidad

- **RAM >90%**: ⚠️ Cerca del límite, reducir batch/grid
- **Tiempo restante <10%**: 🔴 Se acerca al límite, guardará automáticamente

---

## 💾 Política de Checkpoints

### Smart Save (automático)

El notebook usa `QC_Trainer_v4` con sistema inteligente:

1. **Mejores N modelos** (default: 5)
   - Solo guarda si mejora métricas
   - Borra automáticamente checkpoints antiguos peores

2. **Último modelo** (siempre)
   - `last_model.pth` - checkpoint más reciente
   - Permite continuar entrenamiento

3. **Checkpoints periódicos**
   - Local: Cada `SAVE_EVERY_EPISODES` (default: 10)
   - Drive: Cada `DRIVE_SYNC_EVERY` (default: 50)

### Estructura de Archivos

```
/MyDrive/Atheria/
├── checkpoints/
│   └── {EXPERIMENT_NAME}/
│       ├── best_model.pth          # Mejor modelo
│       ├── best_model_FINAL.pth    # Copia al finalizar
│       ├── last_model.pth          # Último checkpoint
│       └── checkpoint_ep*.pth      # Checkpoints históricos
├── logs/
│   └── {EXPERIMENT_NAME}/
│       ├── training_log_*.json     # Log de entrenamiento
│       ├── training_summary.png    # Gráficos
│       └── {EXPERIMENT_NAME}_REPORT.md  # Reporte final
└── exports/
    └── {EXPERIMENT_NAME}_model.pt  # TorchScript exportado
```

---

## 🔧 Troubleshooting

### Problema: "Drive sync muy lento"

**Solución:**
- Aumentar `DRIVE_SYNC_EVERY` a 100 o más
- Checkpoints locales siguen funcionando
- Sincronizar manualmente cada 2-3 horas

### Problema: "Se quedó sin RAM"

**Síntomas:** Kernel crashed, OOM error

**Solución:**
```python
"GRID_SIZE_TRAINING": 32,  # Reducir de 64 a 32
"hidden_channels": 32,     # Reducir de 64 a 32
```

### Problema: "GPU utilization muy baja (<50%)"

**Solución:**
```python
"GRID_SIZE_TRAINING": 128,  # Aumentar complejidad
"hidden_channels": 128,
```

### Problema: "Checkpoint no se encontró en Drive"

**Verificar:**
1. Drive está montado correctamente
2. Carpeta `/MyDrive/Atheria/checkpoints/` existe
3. Nombre del experimento coincide exactamente

**Solución manual:**
```python
# Buscar manualmente
!ls "/content/drive/MyDrive/Atheria/checkpoints/{EXPERIMENT_NAME}/"
```

### Problema: "Timeout antes de completar"

**Prevención:**
- `MAX_TRAINING_HOURS` debe ser **menor** que límite de Colab/Kaggle
- Colab Free: usar `MAX_TRAINING_HOURS=10`
- Colab Pro: usar `MAX_TRAINING_HOURS=20`
- El notebook guarda automáticamente antes de timeout

---

## 💡 Mejores Prácticas

### 1. Sesiones Múltiples (Colab Free)

Si tienes límite de 12h/día:
- **Sesión 1**: 10h (ej: 9am - 7pm)
- **Sesión 2**: 10h (ej: 9am - 7pm siguiente día)
- Auto-resume conecta ambas sesiones

### 2. Monitoring Externo (opcional)

Para saber cuándo termina sin estar pendiente:

**Opción A: Email con Colab**
```python
# Agregar al final del training loop
from google.colab import auth
# Configurar para enviar email al completar
```

**Opción B: Revisar Drive**
- Revisar carpeta de Drive cada 2-3 horas
- Verificar timestamp de `last_model.pth`

### 3. Validación Periódica

Cada 200-300 episodios:
1. Pausar entrenamiento (Ctrl+C)
2. Cargar mejor modelo
3. Ejecutar inferencia de prueba
4. Si resultados buenos → continuar
5. Si no mejora → ajustar learning rate

### 4. Cuota de GPU

**Colab Free:**
- Usar en horarios de baja demanda (madrugada)
- No abusar: respetar límites de uso justo

**Kaggle:**
- Aprovechar 30h/semana completas
- Planificar 3 sesiones de 10h

---

## 📈 Estimar Tiempo de Entrenamiento

**Fórmula aproximada:**
```
Tiempo (horas) = (TOTAL_EPISODES × QCA_STEPS × GRID_SIZE²) / (GPU_SPEED × 3600)
```

Donde:
- `GPU_SPEED` (T4) ≈ 50,000,000 células/segundo

**Ejemplos:**

| Grid | Episodes | QCA Steps | Tiempo (T4) |
|------|----------|-----------|-------------|
| 32   | 500      | 100       | ~1h         |
| 64   | 500      | 100       | ~4h         |
| 64   | 1000     | 100       | ~8h         |
| 128  | 1000     | 100       | ~32h        |

---

## 🎓 Ejemplos Completos

### Ejemplo 1: Primera Prueba (2 horas)

```python
EXPERIMENT_CONFIG = {
    "EXPERIMENT_NAME": "Test_UNET_First",
    "MODEL_ARCHITECTURE": "UNET",
    "MODEL_PARAMS": {"d_state": 4, "hidden_channels": 16},
    "GRID_SIZE_TRAINING": 32,
    "QCA_STEPS_TRAINING": 50,
    "TOTAL_EPISODES": 100,
    "SAVE_EVERY_EPISODES": 5,
    "DRIVE_SYNC_EVERY": 20,
    "MAX_TRAINING_HOURS": 2,
    "AUTO_RESUME": False,  # Primera vez
}
```

### Ejemplo 2: Experimento Serio (12 horas)

```python
EXPERIMENT_CONFIG = {
    "EXPERIMENT_NAME": "UNET_Production_v1",
    "MODEL_ARCHITECTURE": "UNET",
    "MODEL_PARAMS": {"d_state": 8, "hidden_channels": 64},
    "GRID_SIZE_TRAINING": 64,
    "QCA_STEPS_TRAINING": 100,
    "TOTAL_EPISODES": 800,
    "SAVE_EVERY_EPISODES": 10,
    "DRIVE_SYNC_EVERY": 50,
    "MAX_TRAINING_HOURS": 11,  # Margen de seguridad
    "AUTO_RESUME": True,
}
```

### Ejemplo 3: Continuar Entrenamiento

```python
# Mismo EXPERIMENT_NAME que antes
EXPERIMENT_CONFIG = {
    "EXPERIMENT_NAME": "UNET_Production_v1",  # ⚠️ Mismo nombre
    "MODEL_ARCHITECTURE": "UNET",
    "MODEL_PARAMS": {"d_state": 8, "hidden_channels": 64},
    "GRID_SIZE_TRAINING": 64,
    "QCA_STEPS_TRAINING": 100,
    "TOTAL_EPISODES": 1500,  # Aumentado de 800 a 1500
    "SAVE_EVERY_EPISODES": 10,
    "DRIVE_SYNC_EVERY": 50,
    "MAX_TRAINING_HOURS": 11,
    "AUTO_RESUME": True,  # ✅ Clave: auto-resume activado
}
```

---

## 🎯 Conclusión

El notebook `Atheria_Progressive_Training.ipynb` está optimizado para:
- ✅ Entrenamientos largos sin supervisión
- ✅ Aprovechamiento máximo de cuota de GPU
- ✅ Auto-recuperación robusta
- ✅ Monitoreo de recursos en tiempo real
- ✅ Gestión inteligente de checkpoints

**Workflow simple:**
1. Configurar experimento
2. Ejecutar todas las celdas
3. Dejar corriendo
4. Repetir si necesario (auto-resume)

**¡Feliz entrenamiento! 🚀**
