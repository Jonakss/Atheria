# Visualizaciones de Campos Cuánticos - Métricas Bar

## 📋 Resumen

Sistema de widgets colapsables en la barra inferior (`MetricsBar`) que muestran métricas críticas de la simulación con visualizaciones en tiempo real de campos cuánticos.

## 🎯 Widgets Implementados

### 1. **Energía de Vacío** (Vacuum Energy)
- **Tipo de Campo:** `energy` / `density`
- **Datos:** `simData.map_data` (densidad |ψ|²)
- **Visualización:** Mini gráfico de línea mostrando distribución de energía
- **Métrica:** Promedio de |ψ|² multiplicado por factor de conversión (0.0042)
- **Unidad:** EV (Energía de Vacío)
- **Estado:** `good` (verde)

### 2. **Entropía Local** (Local Entropy)
- **Tipo de Campo:** `density`
- **Datos:** `simData.map_data` (distribución de probabilidad)
- **Visualización:** Mini gráfico de línea mostrando distribución de entropía (Shannon)
- **Cálculo:** H = -Σ p_i * log2(p_i) donde p_i es probabilidad normalizada
- **Unidad:** BITS
- **Estado:** `neutral` (gris)

### 3. **Simetría IONQ** (IONQ Symmetry)
- **Tipo de Campo:** `phase`
- **Datos:** `simData.map_data` (simetría espacial)
- **Visualización:** Mini visualización de fase (color cíclico HSV)
- **Cálculo:** Simetría horizontal + vertical promedio (reflexión sobre ejes)
- **Unidad:** IDX (Índice de Simetría)
- **Estado:** `good` (verde)

### 4. **Decaimiento** (Decay Rate)
- **Tipo de Campo:** `flow`
- **Datos:** `simData.flow_data.magnitude` (magnitud de flujo como proxy)
- **Visualización:** Mini visualización de flujo (gradiente de color)
- **Cálculo:** Gamma decay rate convertido a rad/s (factor 0.012)
- **Unidad:** RAD/S
- **Estado:** `warning` (ámbar)

## 🔄 Estados del Widget

### Colapsado
- **Apariencia:** Solo nombre verticalmente rotado (-90°)
- **Tamaño:** Mínimo, solo texto
- **Interacción:** Click para expandir
- **Hover:** Highlight sutil (bg-white/5)

### Expandido
- **Apariencia:** 
  - Métrica completa (`MetricItem`): Label + Value + Unit
  - Mini visualización del campo (64x48px)
  - Estadísticas superpuestas (valor actual)
- **Interacción:** Click para colapsar
- **Hover:** Highlight sutil

## 📊 Tipos de Visualización de Campos

### 1. **Densidad/Energía** (`density` / `energy`)
- **Formato:** Gráfico de línea (polyline SVG)
- **Color:** Verde (good), Ámbar (warning), Azul (neutral)
- **Características:**
  - Muestra distribución muestreada (64 puntos)
  - Línea promedio punteada (opcional)
  - Valor actual en esquina inferior derecha

### 2. **Flujo** (`flow`)
- **Formato:** Gradiente de color (rectángulos SVG)
- **Color:** Azul (rgba(59, 130, 246, alpha))
- **Características:**
  - Alpha basado en magnitud normalizada
  - Muestra dirección y magnitud del flujo

### 3. **Fase** (`phase`)
- **Formato:** Color cíclico HSV (rectángulos SVG)
- **Color:** Espectro completo (hue rotando 0-360°)
- **Características:**
  - Saturation: 70%
  - Lightness: 50%
  - Muestra distribución de fase cuántica

## 🔧 Optimizaciones de Rendimiento

### Muestreo de Datos
- **Máximo de puntos:** 64 (para rendimiento)
- **Método:** Sampling uniforme con step = `length / 64`
- **Normalización:** A [0, 1] para consistencia visual

### Actualización
- **Memoización:** `useMemo` para cálculos de visualización
- **Re-renderizado:** Solo cuando `fieldData` o `isCollapsed` cambian
- **Debounce:** Implícito vía React state updates

## 🚀 Extensiones Futuras

### Campos Adicionales Disponibles

1. **Campo Real/Imaginario** (`complex_3d_data`)
   - `real`: Parte real de ψ
   - `imag`: Parte imaginaria de ψ
   - Visualización: Diagrama de Argand o proyecciones 2D

2. **Campo de Fase HSV** (`phase_hsv_data`)
   - `hue`: Matiz de fase (0-360°)
   - `saturation`: Saturación basada en densidad
   - `value`: Valor basado en magnitud
   - Visualización: Mini mapa de color completo

3. **Campo de Flujo Vectorial** (`flow_data`)
   - `dx`, `dy`: Componentes X e Y del flujo
   - `magnitude`: Magnitud del vector
   - Visualización: Mini campo vectorial (flechas)

4. **Coordenadas Poincaré** (`poincare_coords`)
   - Sección de Poincaré del espacio de fases
   - Visualización: Mini scatter plot 2D

### Nuevos Widgets Potenciales

1. **Campo de Fuerza** (Force Field)
   - Gradiente de energía potencial
   - Visualización: Campo vectorial direccional

2. **Campo de Correlación** (Correlation Field)
   - Correlación espacial entre partículas
   - Visualización: Heatmap de correlación

3. **Campo de Coherencia** (Coherence Field)
   - Coherencia cuántica local
   - Visualización: Mapa de coherencia (0-1)

## 📝 Notas de Implementación

### Datos Disponibles en `simData`
```typescript
interface SimData {
  map_data?: number[][];              // Densidad |ψ|²
  flow_data?: {
    dx: number[][];
    dy: number[][];
    magnitude?: number[][];
  };
  phase_hsv_data?: {
    hue: number[][];
    saturation: number[][];
    value: number[][];
  };
  complex_3d_data?: {
    real: number[][];
    imag: number[][];
  };
  poincare_coords?: number[][];
  // ... otros campos
}
```

### Colapsado Global vs. Individual
- **Global (`expanded`):** Controla si la barra completa está expandida
- **Individual (`collapsedWidgets`):** Set de IDs de widgets colapsados
- **Comportamiento:** Widget individual puede estar colapsado incluso si la barra está expandida

## 🔗 Referencias

- `frontend/src/modules/Dashboard/components/MetricsBar.tsx`: Implementación principal
- `frontend/src/modules/Dashboard/components/FieldWidget.tsx`: Componente de widget colapsable
- `frontend/src/modules/Dashboard/components/MetricItem.tsx`: Componente base de métrica
- `src/pipelines/pipeline_viz.py`: Generación de datos de visualización backend

---

*Última actualización: 2024-12-XX*

