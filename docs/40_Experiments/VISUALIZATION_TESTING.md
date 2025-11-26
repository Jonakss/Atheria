# Guía de Pruebas por Visualización

Este documento explica cómo probar y qué buscar en cada visualización.

---

## Visualizaciones Básicas

### 1. Density (Densidad |ρ|²)

**Qué muestra:** Probabilidad de encontrar "algo" en cada posición.

**Cómo probar:**
1. Cargar cualquier modelo
2. Seleccionar "Densidad |ρ|²"
3. Iniciar simulación
4. Observar durante 30-60 segundos

**Qué buscar:**
- ✅ **Zonas brillantes** = alta densidad (partículas, estructuras)
- ✅ **Zonas oscuras** = baja densidad (vacío)
- ✅ **Patrones** = estructuras emergentes
- ✅ **Cambios lentos** = evolución suave
- ✅ **Cambios rápidos** = dinámica activa

**Experimentos sugeridos:**
- **Sistema cerrado (GAMMA_DECAY=0):** ¿Se conserva la densidad total?
- **Sistema abierto (GAMMA_DECAY>0):** ¿Decae la densidad total?
- **Comparar modelos:** ¿MLP vs UNet crean diferentes patrones?

**Interpretación:**
- Si la densidad se concentra en zonas = estructuras emergentes
- Si la densidad es uniforme = sistema sin estructura
- Si la densidad oscila = posible oscilador

---

### 2. Phase (Fase - Arg)

**Qué muestra:** Argumento (ángulo) del estado cuántico complejo.

**Cómo probar:**
1. Seleccionar "Fase (Arg)"
2. Observar colores (cada color = fase diferente)
3. Buscar patrones de color

**Qué buscar:**
- ✅ **Patrones de color** = estructuras con fase coherente
- ✅ **Cambios graduales** = ondas, rotaciones
- ✅ **Cambios abruptos** = discontinuidades, bordes
- ✅ **Oscilaciones** = fase cambiando periódicamente

**Experimentos sugeridos:**
- **Buscar ondas:** ¿Hay patrones que se propagan?
- **Buscar vórtices:** ¿Hay espirales de color?
- **Comparar con density:** ¿Las zonas de alta densidad tienen fase coherente?

**Interpretación:**
- Fase coherente = estructura organizada
- Fase caótica = ruido
- Fase oscilante = oscilador

---

### 3. Energy (Energía)

**Qué muestra:** Energía total por posición (suma de |ψ|² sobre canales).

**Cómo probar:**
1. Seleccionar "Energía"
2. Observar distribución
3. Comparar con density (deberían ser similares)

**Qué buscar:**
- ✅ **Conservación:** En sistema cerrado, energía total debería ser constante
- ✅ **Decaimiento:** En sistema abierto, energía debería decaer
- ✅ **Distribución:** ¿Dónde se concentra la energía?

**Experimentos sugeridos:**
- **Sistema cerrado:** Medir energía total inicial vs después de 1000 pasos
- **Sistema abierto:** Verificar que decae exponencialmente
- **Comparar modelos:** ¿Cuál conserva mejor la energía?

**Interpretación:**
- Energía constante = sistema cerrado funcionando correctamente
- Energía decreciente = decaimiento (GAMMA_DECAY)
- Energía creciente = error (no debería pasar)

---

## Visualizaciones Avanzadas

### 4. Entropy (Entropía - Complejidad)

**Qué muestra:** Complejidad/información por célula (entropía de Shannon).

**Cómo probar:**
1. Seleccionar "Entropía (Complejidad)"
2. Observar: zonas brillantes = alta complejidad
3. Comparar al inicio vs después de tiempo

**Qué buscar:**
- ✅ **Alta entropía** = caos, ruido, alta información
- ✅ **Baja entropía** = orden, estructuras simples
- ✅ **Balance** = ideal para A-Life (ni demasiado orden ni caos)
- ✅ **Evolución:** ¿Aumenta o disminuye con el tiempo?

**Experimentos sugeridos:**
- **Medir evolución:** Entropía promedio al inicio vs después de 1000 pasos
- **Buscar balance:** Zonas con entropía media (ni muy alta ni muy baja)
- **Comparar modelos:** ¿Cuál genera más complejidad?

**Interpretación:**
- Entropía creciente = sistema ganando complejidad (bueno para A-Life)
- Entropía decreciente = sistema ordenándose (puede ser bueno o malo)
- Balance orden/complejidad = zona donde puede emerger vida

**Criterio de éxito:**
- Si la entropía aumenta moderadamente = sistema explorando espacio de estados
- Si la entropía se mantiene estable = sistema en equilibrio

---

### 5. Coherence (Coherencia - Estructuras)

**Qué muestra:** Sincronización de fase entre células vecinas.

**Cómo probar:**
1. Seleccionar "Coherencia (Estructuras)"
2. Observar: zonas brillantes = alta coherencia
3. **CRÍTICO:** Buscar zonas que se mueven = gliders

**Qué buscar:**
- ✅ **Alta coherencia** = estructuras organizadas (gliders, ondas, vórtices)
- ✅ **Baja coherencia** = ruido, caos
- ✅ **Patrones que se mueven** = **GLIDERS** (estructuras vivas)
- ✅ **Patrones estacionarios** = osciladores, still lifes

**Experimentos sugeridos:**
- **Buscar gliders:** Zonas de alta coherencia que se mueven
- **Medir estabilidad:** ¿Las estructuras persisten?
- **Comparar con flow:** ¿Las zonas de alta coherencia tienen flujo direccional?

**Interpretación:**
- Alta coherencia + movimiento = **GLIDER** (¡A-Life!)
- Alta coherencia + estacionario = oscilador o still life
- Baja coherencia = ruido, no hay estructuras

**Criterio de éxito:**
- Si encuentras zonas de alta coherencia que se mueven = **¡GLIDER DETECTADO!**
- Si las estructuras persisten > 100 pasos = estructura estable

---

### 6. Channel Activity (Actividad por Canal)

**Qué muestra:** Qué canales del estado cuántico están más activos.

**Cómo probar:**
1. Seleccionar "Actividad por Canal"
2. Observar: cada posición muestra el canal dominante
3. Buscar patrones de especialización

**Qué buscar:**
- ✅ **Especialización:** ¿Algunos canales dominan en ciertas zonas?
- ✅ **Uniformidad:** ¿Todos los canales tienen actividad similar?
- ✅ **Patrones espaciales:** ¿Hay zonas donde ciertos canales dominan?

**Experimentos sugeridos:**
- **Con d_state=8:** ¿Canales 0-3 vs 4-7 tienen roles diferentes?
- **Buscar canales muertos:** ¿Hay canales siempre oscuros?
- **Buscar especialización:** ¿Zonas donde un canal específico domina?

**Interpretación:**
- Especialización = canales con roles distintos (bueno para complejidad)
- Uniformidad = todos los canales iguales (menos interesante)
- Canales muertos = posible problema (canal no se usa)

**Criterio de éxito:**
- Si hay especialización = sistema aprendiendo a usar diferentes "dimensiones"
- Si todos los canales están activos = buen uso del espacio de estados

---

### 7. Physics (Física - Matriz A)

**Qué muestra:** "Fuerza" de la interacción física local.

**Cómo probar:**
1. Seleccionar "Física (Matriz A)"
2. Observar: zonas brillantes = alta "fuerza física"
3. Comparar con otras visualizaciones

**Qué buscar:**
- ✅ **Zonas brillantes** = donde la Ley M está más activa
- ✅ **Zonas oscuras** = donde hay poca transformación
- ✅ **Correlación:** ¿Las estructuras tienen alta física?

**Experimentos sugeridos:**
- **Comparar con coherence:** ¿Las zonas de alta coherencia tienen alta física?
- **Comparar con density:** ¿Las zonas de alta densidad tienen alta física?
- **Buscar patrones:** ¿Hay zonas donde la física es especialmente alta?

**Interpretación:**
- Alta física = transformación activa (donde ocurre la "magia")
- Baja física = zonas quietas
- Correlación con estructuras = la física está creando las estructuras

---

### 8. Flow (Flujo - Quiver)

**Qué muestra:** Campo vectorial de delta_psi (dirección de cambio).

**Cómo probar:**
1. Seleccionar "Flujo (Quiver)"
2. Observar flechas: dirección y magnitud
3. Buscar patrones de movimiento

**Qué buscar:**
- ✅ **Flechas largas** = cambios grandes
- ✅ **Flechas en espiral** = vórtices
- ✅ **Flechas paralelas** = ondas o flujo direccional
- ✅ **Movimiento de estructuras** = gliders

**Experimentos sugeridos:**
- **Buscar gliders:** Flechas paralelas que se mueven juntas
- **Buscar vórtices:** Flechas en espiral
- **Medir velocidad:** Longitud de flechas = velocidad de cambio

**Interpretación:**
- Flujo direccional = movimiento organizado (gliders, ondas)
- Flujo caótico = cambios aleatorios
- Flujo en espiral = vórtices, rotaciones

---

### 9. Spectral (Espectro - FFT)

**Qué muestra:** Transformada de Fourier espacial (frecuencias espaciales).

**Cómo probar:**
1. Seleccionar "Espectro (FFT)"
2. Observar: patrones = frecuencias dominantes
3. Buscar simetrías

**Qué buscar:**
- ✅ **Puntos brillantes** = frecuencias espaciales dominantes
- ✅ **Patrones simétricos** = estructuras periódicas
- ✅ **Centro brillante** = componente DC (promedio)

**Experimentos sugeridos:**
- **Buscar periodicidad:** ¿Hay frecuencias dominantes?
- **Comparar con density:** ¿Las estructuras tienen frecuencias específicas?
- **Buscar ondas:** Frecuencias en anillos = ondas

**Interpretación:**
- Frecuencias específicas = estructuras periódicas
- Espectro plano = ruido
- Anillos = ondas con longitud de onda específica

---

## Combinaciones de Visualizaciones

### Para Detectar Gliders
1. **Coherence** - Buscar zonas brillantes
2. **Flow** - Verificar que hay flujo direccional
3. **Density** - Verificar que hay estructura visible

### Para Detectar Osciladores
1. **Phase** - Buscar cambios periódicos
2. **Coherence** - Verificar alta coherencia
3. **Spectral** - Buscar frecuencias específicas

### Para Entender el Sistema
1. **Entropy** - Ver complejidad general
2. **Coherence** - Ver estructuras
3. **Physics** - Ver dónde ocurre la transformación
4. **Channel Activity** - Ver uso de canales

---

## Checklist de Pruebas

Para cada visualización:
- [ ] ¿Se carga correctamente?
- [ ] ¿Muestra datos válidos?
- [ ] ¿Hay patrones visibles?
- [ ] ¿Los patrones tienen sentido físico?
- [ ] ¿Se actualiza correctamente?
- [ ] ¿Es útil para tu objetivo?

---

## Troubleshooting

### "La visualización está en blanco"
- Verificar que hay un modelo cargado
- Verificar que la simulación está corriendo
- Verificar que hay datos en map_data

### "La visualización no se actualiza"
- Verificar que la simulación no está pausada
- Verificar conexión WebSocket
- Verificar logs del servidor

### "Los colores no tienen sentido"
- Algunas visualizaciones usan normalización automática
- Los colores son relativos (no absolutos)
- Comparar con otras visualizaciones para contexto

