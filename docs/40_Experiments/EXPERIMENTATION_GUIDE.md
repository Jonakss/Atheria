# Guía de Experimentación con Aetheria

## Índice
1. [Guía de Aprendizaje Progresivo](#guía-de-aprendizaje-progresivo)
2. [Cómo Probar Cada Visualización](#cómo-probar-cada-visualización)
3. [Estrategias de Experimentación](#estrategias-de-experimentación)
4. [Optimizaciones y Eficiencias](#optimizaciones-y-eficiencias)

---

## Guía de Aprendizaje Progresivo

### Fase 1: Fundamentos (Semanas 1-2)
**Objetivo:** Entender la física básica y las visualizaciones fundamentales.

#### Experimento 1.1: Sistema Cerrado Simple
```yaml
Modelo: MLP
d_state: 4
grid_size: 32
GAMMA_DECAY: 0.0  # Sin decaimiento
Visualizaciones a probar:
  - density: Ver distribución de energía
  - phase: Ver fases del estado cuántico
  - energy: Ver energía total
```

**Qué observar:**
- ¿Se conserva la energía? (debería ser constante en sistema cerrado)
- ¿Hay patrones en la fase?
- ¿La densidad se distribuye uniformemente o forma estructuras?

#### Experimento 1.2: Sistema Abierto (Lindblad)
```yaml
Modelo: MLP
d_state: 4
grid_size: 32
GAMMA_DECAY: 0.01  # Con decaimiento
Visualizaciones:
  - density: Ver cómo decae la energía
  - physics: Ver la "fuerza" de la Ley M
  - entropy: Ver si gana complejidad
```

**Qué observar:**
- ¿La energía decae? (debería decaer con GAMMA_DECAY > 0)
- ¿La Ley M puede "ganar" contra el decaimiento?
- ¿Aparece complejidad (entropía alta)?

---

### Fase 2: Arquitecturas Avanzadas (Semanas 3-4)
**Objetivo:** Comparar diferentes arquitecturas y entender sus diferencias.

#### Experimento 2.1: Comparación MLP vs UNet
```yaml
Experimento A: MLP
  - d_state: 8
  - hidden_channels: 16
  - grid_size: 64

Experimento B: UNET_UNITARY
  - d_state: 8
  - hidden_channels: 32
  - grid_size: 64

Métricas a comparar:
  - Velocidad de inferencia (FPS)
  - Complejidad emergente (entropy)
  - Estructuras coherentes (coherence)
```

**Qué observar:**
- ¿MLP es más rápido? (sí, ~10-20x)
- ¿UNet crea más estructuras? (probablemente sí)
- ¿Cuál conserva mejor la energía?

#### Experimento 2.2: RMSNorm vs GroupNorm
```yaml
Experimento A: UNET_UNITARY (GroupNorm)
Experimento B: UNET_UNITARY_RMSNORM (RMSNorm)

Comparar:
  - Velocidad
  - Conservación de energía
  - Estabilidad numérica
```

**Qué observar:**
- ¿RMSNorm es más rápido? (sí, ~15-20%)
- ¿Mejor conservación de energía?
- ¿Más estable numéricamente?

---

### Fase 3: Memoria Temporal (Semanas 5-6)
**Objetivo:** Entender cómo la memoria temporal afecta la evolución.

#### Experimento 3.1: ConvLSTM vs UNet
```yaml
Experimento A: UNET_UNITARY
Experimento B: UNET_CONVLSTM

Visualizaciones clave:
  - coherence: ¿Más estructuras coherentes?
  - entropy: ¿Más complejidad?
  - channel_activity: ¿Canales especializados?
```

**Qué observar:**
- ¿ConvLSTM crea patrones más complejos?
- ¿Hay "memoria" de eventos pasados?
- ¿Aparecen osciladores o ritmos?

---

### Fase 4: Análisis Profundo (Semanas 7+)
**Objetivo:** Usar todas las herramientas para entender el sistema.

#### Experimento 4.1: Búsqueda de A-Life
```yaml
Configuración:
  - Modelo: UNET_CONVLSTM
  - d_state: 8
  - GAMMA_DECAY: 0.01
  - grid_size: 128

Visualizaciones a usar:
  1. coherence: Buscar zonas de alta coherencia (estructuras)
  2. entropy: Buscar balance orden/complejidad
  3. physics: Ver dónde la Ley M es más activa
  4. flow: Ver movimiento de estructuras
  5. t-SNE: Analizar evolución temporal

Métricas de éxito:
  - Estructuras que se mueven (gliders)
  - Osciladores estables
  - Replicación/crecimiento
```

---

## Cómo Probar Cada Visualización

### 1. Density (Densidad |ρ|²)
**Qué muestra:** Distribución de probabilidad del estado cuántico.

**Cómo probar:**
1. Cargar cualquier modelo
2. Iniciar simulación
3. Seleccionar "Densidad"
4. Observar: ¿Hay concentraciones? ¿Se distribuye uniformemente?

**Qué buscar:**
- Zonas brillantes = alta densidad (partículas)
- Zonas oscuras = baja densidad (vacío)
- Patrones = estructuras emergentes

---

### 2. Phase (Fase)
**Qué muestra:** Argumento (fase) del estado cuántico complejo.

**Cómo probar:**
1. Seleccionar "Fase (Arg)"
2. Observar colores: cada color = fase diferente

**Qué buscar:**
- Patrones de color = estructuras con fase coherente
- Cambios rápidos = oscilaciones
- Zonas uniformes = fase constante

---

### 3. Entropy (Entropía)
**Qué muestra:** Complejidad/información por célula (entropía de Shannon).

**Cómo probar:**
1. Seleccionar "Entropía (Complejidad)"
2. Observar: zonas brillantes = alta complejidad

**Qué buscar:**
- **Alta entropía** = caos, ruido, alta información
- **Baja entropía** = orden, estructuras simples
- **Balance** = ideal para A-Life (ni demasiado orden ni demasiado caos)

**Experimento sugerido:**
- Comparar entropía al inicio vs después de 1000 pasos
- ¿Aumenta? = sistema gana complejidad
- ¿Disminuye? = sistema se ordena

---

### 4. Coherence (Coherencia)
**Qué muestra:** Coherencia de fase entre células vecinas.

**Cómo probar:**
1. Seleccionar "Coherencia (Estructuras)"
2. Observar: zonas brillantes = alta coherencia

**Qué buscar:**
- **Alta coherencia** = estructuras organizadas (gliders, ondas, vórtices)
- **Baja coherencia** = ruido, caos
- **Patrones** = estructuras emergentes

**Experimento sugerido:**
- Buscar zonas de alta coherencia que se mueven = gliders
- Zonas de alta coherencia estacionarias = osciladores

---

### 5. Channel Activity (Actividad por Canal)
**Qué muestra:** Qué canales del estado cuántico están más activos.

**Cómo probar:**
1. Seleccionar "Actividad por Canal"
2. Observar: cada posición muestra el canal dominante

**Qué buscar:**
- **Especialización:** ¿Algunos canales son más activos en ciertas zonas?
- **Uniformidad:** ¿Todos los canales tienen actividad similar?
- **Patrones:** ¿Hay zonas donde ciertos canales dominan?

**Experimento sugerido:**
- Si d_state=8, ver si canales 0-3 vs 4-7 tienen roles diferentes
- ¿Hay canales "muertos" (siempre oscuros)?

---

### 6. Physics (Física - Matriz A)
**Qué muestra:** "Fuerza" de la interacción física local.

**Cómo probar:**
1. Seleccionar "Física (Matriz A)"
2. Observar: zonas brillantes = alta "fuerza física"

**Qué buscar:**
- **Zonas brillantes** = donde la Ley M está más activa
- **Zonas oscuras** = donde hay poca transformación
- **Correlación con estructuras:** ¿Las estructuras vivas tienen alta física?

---

### 7. Flow (Flujo)
**Qué muestra:** Campo vectorial de delta_psi (dirección de cambio).

**Cómo probar:**
1. Seleccionar "Flujo (Quiver)"
2. Observar flechas: dirección y magnitud del cambio

**Qué buscar:**
- **Flechas largas** = cambios grandes
- **Flechas en espiral** = vórtices
- **Flechas paralelas** = ondas o flujo direccional
- **Movimiento de estructuras** = gliders

---

### 8. Spectral (Espectro FFT)
**Qué muestra:** Transformada de Fourier espacial (frecuencias espaciales).

**Cómo probar:**
1. Seleccionar "Espectro (FFT)"
2. Observar: patrones = frecuencias dominantes

**Qué buscar:**
- **Puntos brillantes** = frecuencias espaciales dominantes
- **Patrones simétricos** = estructuras periódicas
- **Centro brillante** = componente DC (promedio)

---

## Estrategias de Experimentación

### Estrategia 1: Búsqueda Sistemática
1. **Fijar arquitectura** (ej. UNET_UNITARY)
2. **Variar un parámetro** a la vez:
   - d_state: 4, 8, 16
   - GAMMA_DECAY: 0.0, 0.01, 0.1
   - grid_size: 32, 64, 128
3. **Medir métricas:**
   - Entropía promedio
   - Coherencia máxima
   - Velocidad (FPS)
4. **Documentar resultados**

### Estrategia 2: Búsqueda de A-Life
1. **Usar visualizaciones clave:**
   - Coherence: buscar estructuras
   - Flow: buscar movimiento
   - Entropy: buscar balance
2. **Guardar historia** cuando veas algo interesante
3. **Analizar con t-SNE:**
   - Universe Atlas: ver evolución
   - Cell Chemistry: ver tipos de células
4. **Iterar:** ajustar parámetros y repetir

### Estrategia 3: Comparación de Arquitecturas
1. **Entrenar mismo experimento** con diferentes modelos:
   - MLP
   - UNET_UNITARY
   - UNET_UNITARY_RMSNORM
   - UNET_CONVLSTM
2. **Comparar métricas:**
   - Velocidad
   - Complejidad emergente
   - Estructuras creadas
3. **Elegir mejor arquitectura** para tu objetivo

---

## Optimizaciones y Eficiencias

### Optimizaciones Implementadas

#### 1. Throttling en Pan/Zoom
- **Problema:** Pan/zoom causaba lag
- **Solución:** Throttle a 16ms (~60fps)
- **Resultado:** Pan suave sin lag

#### 2. Guardado de Historia Opcional
- **Problema:** Guardar cada frame consume memoria
- **Solución:** Habilitar solo cuando necesario
- **Uso:** `simulation.enable_history` con `enabled: true`

#### 3. Frame Skip
- **Problema:** Demasiados frames por segundo
- **Solución:** `simulation.set_frame_skip` con `skip: N`
- **Resultado:** Saltar N frames entre visualizaciones

#### 4. Snapshot Sampling
- **Problema:** Snapshots muy frecuentes ralentizan
- **Solución:** Intervalo configurable (default: 500 pasos)
- **Uso:** `simulation.set_snapshot_interval` con `interval: N`

### Optimizaciones Recomendadas

#### 1. Reducir Resolución de Visualización
```python
# En pipeline_viz.py, añadir downsampling opcional
if grid_size > 256:
    map_data = map_data[::2, ::2]  # Reducir a la mitad
```

#### 2. Usar WebWorkers para Cálculos Pesados
- Mover cálculos de t-SNE a WebWorker
- No bloquear UI durante análisis

#### 3. Compresión de Datos
- Comprimir map_data antes de enviar
- Usar formato binario en lugar de JSON

#### 4. Lazy Loading de Visualizaciones
- Solo calcular visualización cuando se selecciona
- Cachear resultados

---

## Checklist de Experimentación

### Antes de Empezar
- [ ] Entender qué muestra cada visualización
- [ ] Configurar guardado de historia si necesitas análisis posterior
- [ ] Ajustar FPS según tu hardware

### Durante el Experimento
- [ ] Observar múltiples visualizaciones
- [ ] Guardar snapshots cuando veas algo interesante
- [ ] Documentar parámetros usados
- [ ] Guardar historia para análisis posterior

### Después del Experimento
- [ ] Analizar con t-SNE (Universe Atlas, Cell Chemistry)
- [ ] Comparar métricas (entropía, coherencia)
- [ ] Guardar historia si quieres revisar después
- [ ] Documentar hallazgos

---

## Ejemplos de Experimentos

### Experimento: "Búsqueda de Gliders"
```yaml
Objetivo: Encontrar estructuras que se mueven (gliders)

Configuración:
  Modelo: UNET_CONVLSTM
  d_state: 8
  hidden_channels: 32
  GAMMA_DECAY: 0.01
  grid_size: 128

Proceso:
  1. Iniciar simulación
  2. Usar visualización "coherence" para encontrar estructuras
  3. Cambiar a "flow" para ver movimiento
  4. Si ves algo interesante:
     - Capturar snapshot manual
     - Guardar historia
     - Analizar con t-SNE

Criterios de éxito:
  - Estructuras coherentes (coherence alto)
  - Movimiento direccional (flow con flechas paralelas)
  - Persistencia temporal (no desaparecen rápido)
```

### Experimento: "Comparación de Normalizaciones"
```yaml
Objetivo: Comparar RMSNorm vs GroupNorm

Experimentos:
  A: UNET_UNITARY (GroupNorm)
  B: UNET_UNITARY_RMSNORM (RMSNorm)

Métricas:
  - FPS promedio
  - Entropía promedio
  - Coherencia máxima
  - Conservación de energía

Visualizaciones:
  - physics: Ver fuerza de interacción
  - entropy: Ver complejidad
  - energy: Ver conservación

Análisis:
  - ¿Cuál es más rápido?
  - ¿Cuál conserva mejor la energía?
  - ¿Cuál crea más estructuras?
```

---

## Troubleshooting

### Problema: "La simulación se tranca al panear"
**Solución:** Ya implementado throttling. Si persiste, aumentar intervalo de throttle.

### Problema: "Demasiados warnings de charts"
**Solución:** Ya arreglado con minWidth/minHeight. Si persiste, verificar que los contenedores tengan tamaño.

### Problema: "No se pausa"
**Solución:** Verificar logs del servidor. El comando debería llegar. Si no, verificar conexión WebSocket.

### Problema: "Memoria se llena"
**Solución:**
- Deshabilitar historia si no la necesitas
- Reducir max_frames en SimulationHistory
- Aumentar intervalo de snapshots

---

## Próximos Pasos

1. **Probar las 3 nuevas visualizaciones** (Entropy, Coherence, Channel Activity)
2. **Experimentar con diferentes parámetros**
3. **Usar historia para análisis posterior**
4. **Implementar más visualizaciones** según necesidad

