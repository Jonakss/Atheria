# Guía de Aprendizaje Progresivo - Aetheria

## Filosofía: De lo Simple a lo Complejo

Esta guía te lleva desde los conceptos básicos hasta experimentos avanzados de A-Life.

---

## Nivel 1: Fundamentos (Semanas 1-2)

### Objetivo
Entender qué es un QCA (Quantum Cellular Automaton) y cómo funciona la física básica.

### Conceptos Clave
- **Estado Cuántico (ψ):** Vector complejo que describe el estado de cada célula
- **Evolución Unitaria:** Transformación que conserva la norma (energía)
- **Densidad (|ψ|²):** Probabilidad de encontrar "algo" en esa posición

### Experimento 1.1: Sistema Cerrado
```python
# Configuración mínima
MODEL_ARCHITECTURE: "MLP"
d_state: 4
grid_size: 32
GAMMA_DECAY: 0.0  # Sin decaimiento (sistema cerrado)
```

**Tareas:**
1. Cargar modelo MLP
2. Iniciar simulación
3. Observar visualización "density"
4. **Pregunta clave:** ¿Se conserva la energía total?

**Qué aprender:**
- En sistema cerrado (GAMMA_DECAY=0), la energía debería conservarse
- La densidad muestra dónde está "concentrada" la información
- MLP es rápido pero simple (sin contexto espacial)

### Experimento 1.2: Sistema Abierto (Lindblad)
```python
MODEL_ARCHITECTURE: "MLP"
GAMMA_DECAY: 0.01  # Con decaimiento
```

**Tareas:**
1. Comparar con Experimento 1.1
2. Observar cómo decae la energía
3. **Pregunta clave:** ¿La Ley M puede "ganar" contra el decaimiento?

**Qué aprender:**
- GAMMA_DECAY introduce "hambre" (decaimiento)
- La Ley M debe crear estructuras que "ganen" contra el decaimiento
- Esto es la base del "metabolismo" en A-Life

---

## Nivel 2: Visualizaciones (Semanas 2-3)

### Objetivo
Aprender a usar todas las visualizaciones para entender el sistema.

### Visualizaciones Básicas

#### Density (Densidad)
- **Qué es:** |ψ|², probabilidad
- **Qué buscar:** Concentraciones, estructuras
- **Cuándo usar:** Siempre, es la base

#### Phase (Fase)
- **Qué es:** Argumento de ψ (ángulo en plano complejo)
- **Qué buscar:** Patrones de color = fase coherente
- **Cuándo usar:** Para ver oscilaciones, rotaciones

#### Energy (Energía)
- **Qué es:** Suma de |ψ|² sobre todos los canales
- **Qué buscar:** Conservación (sistema cerrado) o decaimiento (abierto)
- **Cuándo usar:** Para verificar física

### Visualizaciones Avanzadas

#### Entropy (Entropía)
- **Qué es:** Complejidad/información (entropía de Shannon)
- **Qué buscar:**
  - Alta = caos, ruido
  - Baja = orden, estructuras simples
  - **Balance = ideal para A-Life**
- **Cuándo usar:** Para medir si el sistema gana complejidad

#### Coherence (Coherencia)
- **Qué es:** Sincronización de fase entre vecinos
- **Qué buscar:**
  - Alta = estructuras organizadas (gliders, ondas)
  - Baja = ruido
- **Cuándo usar:** **CRÍTICO para detectar A-Life**

#### Physics (Física)
- **Qué es:** "Fuerza" de la interacción local
- **Qué buscar:** Dónde la Ley M está más activa
- **Cuándo usar:** Para entender dónde ocurre la "magia"

### Ejercicio: Mapa de Visualizaciones
1. Cargar un modelo
2. Para cada visualización:
   - Seleccionarla
   - Observar 30 segundos
   - Anotar: ¿Qué ves? ¿Qué significa?
3. Comparar: ¿Qué visualizaciones muestran lo mismo? ¿Cuáles son complementarias?

---

## Nivel 3: Arquitecturas (Semanas 3-4)

### Objetivo
Entender las diferencias entre arquitecturas y cuándo usar cada una.

### MLP (Multi-Layer Perceptron)
**Características:**
- ✅ Muy rápido (~10-20x más rápido que UNet)
- ❌ Sin contexto espacial (solo ve la célula actual)
- ❌ No puede crear estructuras complejas

**Cuándo usar:**
- Prototipado rápido
- Sistemas simples
- Cuando la velocidad es crítica

### UNet Unitary
**Características:**
- ✅ Contexto espacial (ve vecinos)
- ✅ Puede crear estructuras complejas
- ⚠️ Más lento que MLP
- ✅ Conserva unitariedad (física correcta)

**Cuándo usar:**
- Búsqueda de estructuras emergentes
- Sistemas que necesitan interacción espacial
- Cuando quieres física unitaria

### UNet Unitary + RMSNorm
**Características:**
- ✅ Todo lo de UNet Unitary
- ✅ ~15-20% más rápido
- ✅ Mejor conservación de energía

**Cuándo usar:**
- Siempre que uses UNet Unitary (mejor opción)
- Cuando quieres mejor rendimiento

### UNet ConvLSTM
**Características:**
- ✅ Todo lo de UNet
- ✅ Memoria temporal (recuerda el pasado)
- ⚠️ Más lento (~20-30x más lento que MLP)
- ✅ Puede crear comportamientos complejos

**Cuándo usar:**
- Búsqueda de A-Life avanzado
- Sistemas que necesitan "memoria"
- Cuando quieres osciladores, ritmos, comportamientos temporales

### Ejercicio: Comparación de Arquitecturas
1. Entrenar mismo experimento con:
   - MLP
   - UNET_UNITARY
   - UNET_CONVLSTM
2. Comparar:
   - Velocidad (FPS)
   - Complejidad (entropía)
   - Estructuras (coherencia)
3. Decidir: ¿Cuál es mejor para tu objetivo?

---

## Nivel 4: Análisis Profundo (Semanas 5-6)

### Objetivo
Usar todas las herramientas para entender y optimizar el sistema.

### Herramientas de Análisis

#### 1. t-SNE: Universe Atlas
**Qué hace:** Analiza la evolución temporal del sistema completo.

**Cómo usar:**
1. Habilitar snapshots (automático cuando solicitas análisis)
2. Ejecutar simulación durante tiempo (captura cada 500 pasos)
3. Click en "Atlas del Universo"
4. Observar: ¿Hay fases distintas? ¿Evolución clara?

**Qué buscar:**
- **Clusters** = fases distintas del sistema
- **Trayectorias** = evolución clara
- **Dispersión** = sistema explorando espacio de estados

#### 2. t-SNE: Cell Chemistry
**Qué hace:** Analiza tipos de células en el estado actual.

**Cómo usar:**
1. Pausar simulación en momento interesante
2. Click en "Mapa Químico"
3. Observar: ¿Hay tipos distintos de células?

**Qué buscar:**
- **Clusters** = tipos distintos de células
- **Transiciones** = células en proceso de cambio
- **Especialización** = células con roles distintos

#### 3. Historia de Simulación
**Qué hace:** Guarda frames para análisis posterior.

**Cómo usar:**
1. Habilitar: `simulation.enable_history` con `enabled: true`
2. Ejecutar simulación
3. Guardar: `simulation.save_history`
4. Cargar después para análisis

**Cuándo usar:**
- Cuando ves algo interesante y quieres analizarlo después
- Para comparar diferentes configuraciones
- Para análisis offline

### Ejercicio: Análisis Completo
1. **Preparación:**
   - Cargar modelo UNET_CONVLSTM
   - Habilitar historia
   - Configurar snapshots (intervalo: 500)

2. **Ejecución:**
   - Iniciar simulación
   - Observar múltiples visualizaciones
   - Capturar snapshots cuando veas algo interesante

3. **Análisis:**
   - Universe Atlas: ¿Cómo evoluciona?
   - Cell Chemistry: ¿Hay tipos de células?
   - Revisar historia: ¿Qué pasó en momentos clave?

4. **Documentación:**
   - Anotar hallazgos
   - Guardar configuraciones exitosas
   - Compartir resultados

---

## Nivel 5: Búsqueda de A-Life (Semanas 7+)

### Objetivo
Encontrar y caracterizar estructuras vivas (gliders, osciladores, replicadores).

### Criterios de A-Life

#### 1. Autonomía
- La estructura se mantiene sin intervención externa
- **Visualización:** coherence (alta), entropy (balance)

#### 2. Movimiento (Gliders)
- Estructura que se mueve de forma direccional
- **Visualización:** flow (flechas paralelas), coherence (se mueve)

#### 3. Oscilación
- Estructura que oscila (ritmo)
- **Visualización:** phase (cambios periódicos), spectral (frecuencias)

#### 4. Replicación
- Estructura que se copia a sí misma
- **Visualización:** density (crecimiento), coherence (múltiples estructuras)

### Estrategia de Búsqueda

#### Paso 1: Exploración Amplia
```yaml
Configuración:
  - Probar diferentes d_state: 4, 8, 16
  - Probar diferentes GAMMA_DECAY: 0.0, 0.01, 0.1
  - Probar diferentes arquitecturas

Visualizaciones:
  - coherence: Buscar zonas brillantes
  - flow: Buscar movimiento
  - entropy: Buscar balance
```

#### Paso 2: Refinamiento
```yaml
Cuando encuentres algo interesante:
  1. Capturar snapshot
  2. Guardar historia
  3. Analizar con t-SNE
  4. Ajustar parámetros ligeramente
  5. Repetir
```

#### Paso 3: Caracterización
```yaml
Para estructuras prometedoras:
  1. Medir velocidad (flow)
  2. Medir estabilidad (coherence temporal)
  3. Medir complejidad (entropy)
  4. Documentar comportamiento
```

### Métricas de Éxito

#### Glider Detectado
- ✅ Coherencia alta y estable
- ✅ Movimiento direccional (flow)
- ✅ Persistencia > 100 pasos
- ✅ Velocidad constante

#### Oscilador Detectado
- ✅ Coherencia alta
- ✅ Fase oscilante (phase cambia periódicamente)
- ✅ Frecuencia estable (spectral)

#### Replicador Detectado
- ✅ Múltiples estructuras similares
- ✅ Crecimiento en número
- ✅ Coherencia entre estructuras

---

## Roadmap de Aprendizaje

### Mes 1: Fundamentos
- [ ] Entender física básica
- [ ] Dominar visualizaciones básicas
- [ ] Comparar arquitecturas simples

### Mes 2: Herramientas
- [ ] Dominar todas las visualizaciones
- [ ] Usar t-SNE para análisis
- [ ] Guardar y analizar historia

### Mes 3: Optimización
- [ ] Encontrar mejores parámetros
- [ ] Optimizar para tu hardware
- [ ] Documentar configuraciones exitosas

### Mes 4+: A-Life
- [ ] Buscar gliders
- [ ] Buscar osciladores
- [ ] Buscar replicadores
- [ ] Caracterizar estructuras encontradas

---

## Recursos Adicionales

### Documentos de Referencia
- `docs/TECHNIQUES_ANALYSIS.md` - Análisis de técnicas avanzadas
- `docs/30_Components/VISUALIZATION_RECOMMENDATIONS.md` - Guía de visualizaciones
- `docs/EXPERIMENTATION_GUIDE.md` - Guía de experimentación

### Comandos Útiles
```python
# Habilitar historia
simulation.enable_history({enabled: true})

# Guardar historia
simulation.save_history({filename: "mi_experimento.json"})

# Capturar snapshot manual
simulation.capture_snapshot({})

# Configurar FPS
simulation.set_fps({fps: 30})

# Configurar velocidad
simulation.set_speed({speed: 2.0})
```

---

## Preguntas Frecuentes

### ¿Por qué MLP es tan rápido?
MLP solo usa convoluciones 1x1 (operaciones punto a punto), sin pooling ni skip connections.

### ¿Cuándo usar ConvLSTM?
Cuando quieres comportamientos temporales: osciladores, ritmos, memoria de eventos pasados.

### ¿Qué visualización es más importante?
**Coherence** - es la mejor para detectar estructuras vivas.

### ¿Cómo sé si encontré A-Life?
Si ves estructuras que:
- Se mueven (gliders)
- Oscilan (osciladores)
- Se replican (replicadores)
- Tienen coherencia alta
- Persisten en el tiempo

¡Entonces tienes A-Life! 🎉

