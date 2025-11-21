# Roadmap: Corrección de Visualización + Documentación de Gaps

**Fecha:** 2025-01-21  
**Prioridad:** 🔴 CRÍTICO - Resolver visualización antes de documentar gaps

---

## 🎯 Objetivo Principal

**CORREGIR LA VISUALIZACIÓN QUE APARECE GRIS** antes de documentar los gaps de Knowledge Base.

---

## 🔴 FASE 1: DIAGNÓSTICO Y CORRECCIÓN URGENTE (Hoy)

### Problema Identificado

**Visualización aparece gris cuando debería mostrar datos:**

1. **Backend normaliza `map_data` a [0, 1]** en `normalize_map_data()`
2. **ShaderCanvas espera datos raw** y normaliza internamente usando `u_minValue` y `u_maxValue`
3. **Doble normalización** o datos ya normalizados pasan por normalización del shader
4. **Motor nativo puede estar vacío** y no inicializarse correctamente
5. **Condiciones de renderizado** pueden estar ocultando datos válidos

### Tareas Críticas

#### 1.1 Verificar Flujo de Datos Backend → Frontend ⚡

**Archivos a revisar:**
- `src/pipelines/viz/core.py` - `get_visualization_data()` normaliza `map_data` a [0, 1]
- `src/pipelines/viz/utils.py` - `normalize_map_data()` retorna [0, 1]
- `src/pipelines/core/simulation_loop.py` - Envía `map_data` en `simulation_frame`
- `frontend/src/context/WebSocketContext.tsx` - Recibe y procesa `simulation_frame`

**Problema detectado:**
- Backend normaliza `map_data` a [0, 1] antes de enviar
- Frontend recibe `map_data` ya normalizado
- ShaderCanvas intenta normalizar de nuevo usando `u_minValue` y `u_maxValue`
- **Resultado:** Datos se normalizan dos veces → visualización incorrecta

**Solución propuesta:**
1. **Opción A (Recomendada):** Backend envía datos raw + metadata (min/max)
   - Cambiar `normalize_map_data()` para que retorne datos raw con metadata
   - Agregar `map_data_min` y `map_data_max` al payload
   - ShaderCanvas usa estos valores para normalizar correctamente

2. **Opción B:** Frontend desnormaliza antes de pasar a shader
   - Mantener normalización en backend
   - ShaderCanvas recibe datos normalizados [0, 1]
   - NO normalizar de nuevo en shader, usar directamente

**Decisión:** Verificar qué espera realmente el shader y corregir la inconsistencia.

#### 1.2 Revisar Preprocesado de Shaders ⚡

**Archivos a revisar:**
- `frontend/src/utils/shaderVisualization.ts` - `createTextureFromData()` 
  - Línea 492: Normaliza datos a [0, 255] para textura
  - Línea 121: Shader espera que textura ya esté normalizada a [0, 1]
  - **PROBLEMA:** Hay doble normalización

**Flujo actual (INCORRECTO):**
1. Backend: `map_data` normalizado a [0, 1]
2. Frontend recibe: `map_data` [0, 1]
3. `createTextureFromData()`: Normaliza de [0, 1] → [0, 255] para textura
4. Shader: Lee textura [0, 255] pero la trata como [0, 1] usando `u_minValue` y `u_maxValue`
5. **Resultado:** Visualización incorrecta

**Solución propuesta:**
- Si backend envía datos normalizados [0, 1]:
  - Shader NO debe usar `u_minValue`/`u_maxValue` (asumir [0, 1])
  - Textura debe almacenar [0, 255] pero shader debe leer como [0, 1]
- Si backend envía datos raw:
  - Agregar `map_data_min` y `map_data_max` al payload
  - Shader usa estos valores para normalizar correctamente
  - Textura normaliza usando estos valores antes de crear

#### 1.3 Verificar Inicialización del Motor Nativo ⚡

**Problema:** Motor nativo puede estar vacío y no inicializarse correctamente.

**Archivos a revisar:**
- `src/pipelines/pipeline_server.py` - `handle_load_experiment()` líneas 1469-1501
- `src/pipelines/handlers/inference_handlers.py` - `handle_play()` líneas 45-88
- `src/engines/native_engine_wrapper.py` - `add_initial_particles()`

**Verificaciones:**
- ✅ ¿Se está inicializando el motor nativo al cargar experimento?
- ✅ ¿Se está inicializando al presionar ejecutar?
- ✅ ¿Los logs muestran "partículas agregadas"?
- ✅ ¿El estado `psi` tiene valores > 1e-10 después de inicializar?

#### 1.4 Verificar Condiciones de Renderizado ⚡

**Archivo:** `frontend/src/components/ui/PanZoomCanvas.tsx`

**Líneas clave:**
- Línea 931: `useShaderRendering && mapData && mapData.length > 0 && mapDataWidth > 0 && mapDataHeight > 0`
- Línea 950: `visibility: (dataToRender?.map_data || simData?.map_data) ? 'visible' : 'hidden'`

**Problemas potenciales:**
- Si `mapData` está presente pero todos los valores son 0.5 (gris medio), se renderiza pero se ve gris
- Si `mapDataWidth === 0` o `mapDataHeight === 0`, no se renderiza

**Verificaciones:**
- ¿`mapData` está presente?
- ¿`mapData.length > 0`?
- ¿`mapData[0]?.length > 0`?
- ¿Los valores son todos 0.5 (fallback de normalización)?

#### 1.5 Logging y Debugging Mejorado ⚡

**Agregar logging detallado:**
1. Backend: Log cuando se normaliza `map_data` (min, max, range)
2. Backend: Log cuando se envía `simulation_frame` (shape, min, max, sample)
3. Frontend: Log cuando se recibe `simulation_frame` (shape, min, max, sample)
4. Frontend: Log cuando ShaderCanvas procesa datos (dataMin, dataMax, textureData range)
5. Frontend: Log errores de WebGL/shader compilation

---

## 🟡 FASE 2: CORRECCIÓN DE IMPLEMENTACIÓN (Próximas 2-3 horas)

### 2.1 Corregir Doble Normalización

**Opción elegida:** **Backend envía datos normalizados + metadata (min/max raw)**

**Cambios requeridos:**

1. **Backend (`src/pipelines/viz/core.py`):**
   - Mantener `normalize_map_data()` pero retornar también min/max raw
   - Agregar `map_data_raw_min` y `map_data_raw_max` al resultado

2. **Backend (`src/pipelines/core/simulation_loop.py`):**
   - Incluir `map_data_raw_min` y `map_data_raw_max` en `frame_payload_raw`

3. **Frontend (`frontend/src/components/ui/ShaderCanvas.tsx`):**
   - Recibir `minValue` y `maxValue` desde props
   - Pasar estos valores a `createTextureFromData()`
   - Shader NO debe normalizar (asumir que textura ya está normalizada)

4. **Frontend (`frontend/src/utils/shaderVisualization.ts`):**
   - `createTextureFromData()`: Si recibe datos ya normalizados [0, 1], NO normalizar de nuevo
   - O usar `minValue`/`maxValue` solo para logging, no para normalización

**Alternativa más simple:** Backend envía datos raw + metadata, frontend normaliza una sola vez.

### 2.2 Verificar Inicialización del Motor

**Mejoras:**
- Asegurar que motor nativo se inicializa AL CARGAR experimento
- Asegurar que motor nativo se inicializa AL PRESIONAR ejecutar
- Logging detallado de valores `psi` antes y después de inicialización
- Verificar que partículas se agregan correctamente

### 2.3 Mejorar Manejo de Datos Vacíos/Uniformes

**Mejoras:**
- Detectar cuando `map_data` está todo uniforme (todos 0.5)
- Intentar reinicializar motor si está vacío
- Mostrar mensaje claro al usuario si motor está vacío
- Logging de advertencia cuando datos son uniformes

---

## 🟢 FASE 3: DOCUMENTACIÓN DE GAPS (Después de corregir visualización)

### 3.1 Documentar Conceptos Técnicos Críticos

**Documentos a crear:**
1. `docs/20_Concepts/LAZY_CONVERSION.md`
   - Qué es lazy conversion
   - Por qué se implementó
   - Cómo funciona
   - Cuándo se usa
   - Trade-offs

2. `docs/20_Concepts/ROI_REGION_OF_INTEREST.md`
   - Qué es ROI
   - Por qué se implementó
   - Cómo funciona
   - Cuándo se activa automáticamente
   - Trade-offs

3. `docs/20_Concepts/DENSE_VS_SPARSE_STATE.md`
   - Diferencia entre estado denso y disperso
   - Cuándo usar cada uno
   - Conversión entre formatos
   - Overhead de conversión

4. `docs/20_Concepts/STATE_STALENESS.md`
   - Qué es state staleness
   - Cómo se detecta
   - Cómo se resuelve
   - Optimizaciones relacionadas

5. `docs/20_Concepts/VISUALIZATION_PIPELINE.md`
   - Flujo completo: psi → map_data → frontend → renderizado
   - Normalización y preprocesado
   - Shaders vs Canvas 2D
   - Optimizaciones aplicadas

### 3.2 Guía de Troubleshooting

**Documento:** `docs/99_Templates/TROUBLESHOOTING_GUIDE.md`

**Secciones:**
1. Problemas de Visualización
   - Pantalla gris (datos vacíos/uniformes)
   - Visualización no se actualiza
   - Errores de shaders
   - Fallback a Canvas 2D

2. Problemas de Motor Nativo
   - Motor no inicializa
   - Segmentation fault
   - Servidor se cierra al cambiar motor
   - Estado vacío

3. Problemas de WebSocket
   - Conexión se cierra
   - Comandos no se procesan
   - Datos no llegan

4. Problemas de Rendimiento
   - FPS muy bajo
   - CPU/GPU alta
   - Memory leaks

### 3.3 Patrones de Código

**Documento:** `docs/30_Components/CODING_PATTERNS.md`

**Patrones a documentar:**
1. Yield periódico al event loop
2. Manejo robusto de errores en cleanup
3. Lazy conversion pattern
4. Normalización de datos
5. Manejo de recursos C++

---

## 📋 Checklist de Verificación

### Para Corregir Visualización

- [ ] Verificar que motor nativo se inicializa al cargar experimento
- [ ] Verificar que motor nativo se inicializa al presionar ejecutar
- [ ] Verificar que `map_data` tiene valores válidos (no todo 0.5)
- [ ] Verificar que `map_data` se envía correctamente desde backend
- [ ] Verificar que `map_data` se recibe correctamente en frontend
- [ ] Verificar que shaders procesan datos correctamente
- [ ] Corregir doble normalización (backend vs shader)
- [ ] Agregar logging detallado en puntos clave
- [ ] Probar con motor nativo
- [ ] Probar con motor Python
- [ ] Probar con diferentes `viz_type`

### Para Documentación

- [ ] Documentar lazy conversion
- [ ] Documentar ROI
- [ ] Documentar Dense vs Sparse
- [ ] Documentar state staleness
- [ ] Crear guía de troubleshooting
- [ ] Documentar patrones de código
- [ ] Actualizar MOCs con nuevos documentos
- [ ] Agregar enlaces cruzados

---

## 🔗 Referencias

- [[00_KNOWLEDGE_BASE_GAPS_ANALYSIS.md]] - Análisis completo de gaps
- [[VISUALIZATION_OPTIMIZATION_ANALYSIS.md]] - Análisis de optimizaciones de visualización
- [[Native_Engine_Core.md]] - Documentación del motor nativo
- `src/pipelines/viz/core.py` - Pipeline de visualización
- `frontend/src/utils/shaderVisualization.ts` - Shaders WebGL
- `frontend/src/components/ui/ShaderCanvas.tsx` - Componente de shader

---

**Última actualización:** 2025-01-21  
**Estado:** 🔴 FASE 1 - Diagnóstico en progreso

