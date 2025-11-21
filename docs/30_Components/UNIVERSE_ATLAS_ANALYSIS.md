# Análisis Atlas del Universo - Visualización de Grafos con t-SNE

## 📋 Resumen

El **Atlas del Universo** es una visualización que analiza la evolución temporal del estado cuántico usando **t-SNE** (t-Distributed Stochastic Neighbor Embedding) para reducir la dimensionalidad y visualizar la estructura del espacio de fases en un grafo de nodos y conexiones.

## 🎯 Objetivo

Visualizar cómo evoluciona el estado cuántico a través del tiempo, agrupando estados similares y mostrando transiciones entre ellos. Esto permite entender:

- **Estructuras recurrentes**: Estados que aparecen múltiples veces
- **Transiciones**: Cómo el sistema evoluciona entre diferentes configuraciones
- **Clusters**: Grupos de estados relacionados
- **Complejidad**: Nodos conectados densamente vs. nodos aislados

## 🔬 Metodología

### 1. Captura de Snapshots
- Se capturan **snapshots** del estado cuántico (`psi`) a intervalos regulares (configurable via `snapshot_interval`)
- Cada snapshot representa un punto en el espacio de estados de alta dimensionalidad

### 2. Compresión Dimensional
- Los estados cuánticos son muy dimensionales (ej: `[channels, height, width]` = `[8, 256, 256]` = 524,288 dimensiones)
- Se usa **PCA** (Principal Component Analysis) primero para reducir a `compression_dim` dimensiones (por defecto: 64)

### 3. Reducción con t-SNE
- **t-SNE** reduce las dimensiones comprimidas a 2D para visualización
- Parámetros configurables:
  - `perplexity`: 30 (por defecto) - Controla el balance entre estructura local y global
  - `n_iter`: 1000 (por defecto) - Número de iteraciones del algoritmo

### 4. Construcción del Grafo
- Cada punto en el espacio 2D de t-SNE se convierte en un **nodo**
- Los **edges** (conexiones) se crean basándose en:
  - Proximidad en el espacio t-SNE (puntos cercanos = conexiones)
  - Secuencia temporal (estados consecutivos están conectados)

## 📊 Interpretación del Grafo

### Estructura del Grafo Visualizado

La visualización muestra:

1. **Nodos (círculos)**:
   - **Tamaño**: Puede representar importancia, energía, o número de conexiones
   - **Color**: Puede codificar energía, entropía, u otra métrica
   - **Posición**: Determinada por t-SNE (estados similares están cerca)

2. **Edges (líneas)**:
   - **Grosor**: Puede representar fuerza de conexión o transición
   - **Longitud**: Distancia en el espacio t-SNE
   - **Densidad**: Alta densidad = región del espacio de fases muy transitada

### Patrones Típicos

- **Clusters densos**: Regiones del espacio de fases donde el sistema pasa mucho tiempo (estados estables)
- **Cadenas lineales**: Transiciones progresivas entre estados (evolución suave)
- **Nodos aislados**: Estados únicos o transiciones rápidas (poco tiempo en esos estados)
- **Hubs (nodos grandes)**: Estados que actúan como "puntos de conexión" (muchas transiciones pasan por ellos)

## 🛠️ Implementación

### Backend (`src/analysis/analysis.py`)

```python
def analyze_universe_atlas(
    psi_snapshots: List[torch.Tensor],
    compression_dim: int = 64,
    perplexity: int = 30,
    n_iter: int = 1000
) -> dict:
    """
    Analiza snapshots del estado cuántico usando t-SNE para crear un atlas del universo.
    
    Returns:
        dict con:
        - 'coords': Lista de coordenadas 2D [x, y] para cada snapshot
        - 'metrics': Métricas del grafo (spread, density, etc.)
        - 'snapshot_indices': Índices de los snapshots usados
    """
```

### Frontend (Recepción de Resultados)

El frontend recibe un mensaje WebSocket `analysis_universe_atlas` con:

```typescript
interface UniverseAtlasResult {
  coords: number[][];           // [[x, y], [x, y], ...] - Coordenadas 2D
  metrics?: {
    spread: number;             // Dispersión del grafo
    density: number;            // Densidad de conexiones
    // ... otras métricas
  };
  snapshot_indices?: number[];  // Índices de snapshots usados
}
```

### Visualización del Grafo

El frontend debe renderizar:
1. **Nodos**: Círculos en las posiciones `coords`
2. **Edges**: Líneas conectando nodos cercanos o consecutivos
3. **Interactividad**: 
   - Hover para mostrar información del snapshot
   - Click para ver el estado cuántico correspondiente
   - Zoom y pan para navegar el grafo

## 📋 Uso

### Activación desde la UI

1. Ejecutar la simulación durante suficiente tiempo para capturar snapshots (mínimo 2)
2. Los snapshots se capturan automáticamente según `snapshot_interval` (por defecto: cada 500 pasos)
3. Ir al menú de análisis y seleccionar "Atlas del Universo"
4. El análisis se ejecuta en background (no bloquea la simulación)
5. El resultado se visualiza como un grafo interactivo

### Parámetros Configurables

- **Compression Dimension** (`compression_dim`): Reducción PCA antes de t-SNE (por defecto: 64)
- **Perplexity**: Balance estructura local/global en t-SNE (por defecto: 30)
- **Iterations** (`n_iter`): Iteraciones de t-SNE (por defecto: 1000)

## 🔗 Relaciones con Otros Componentes

### Snapshots (`snapshot_interval`, `enable_snapshots`)
- El análisis depende de tener snapshots capturados
- Ver: [[MEMORY_MANAGEMENT|Gestión de Memoria]] para límites de snapshots

### Visualizaciones de Campos
- Los snapshots usan los mismos estados que las visualizaciones de campos
- Ver: [[FIELD_VISUALIZATIONS|Visualizaciones de Campos Cuánticos]]

### Análisis de Química Celular
- Similar metodología pero enfocada en tipos de células
- Ver: [[CELL_CHEMISTRY_ANALYSIS|Análisis de Química Celular]]

## 📊 Métricas del Grafo

Las métricas calculadas ayudan a interpretar el grafo:

- **Spread**: Dispersión de los nodos (alto = estados muy diversos)
- **Density**: Densidad de conexiones (alto = muchas transiciones)
- **Clustering**: Número de clusters detectados
- **Hub Count**: Número de nodos con muchas conexiones

## 🚀 Extensiones Futuras

1. **Visualización 3D**: Usar t-SNE con 3 dimensiones para mejor separación
2. **Animación**: Mostrar la evolución temporal del grafo (nodos apareciendo en orden)
3. **Filtrado**: Filtrar nodos por energía, entropía, u otras métricas
4. **Exportación**: Exportar el grafo en formato GraphML, GEXF, o similar
5. **Análisis de Comunidades**: Detectar comunidades en el grafo (Louvain, etc.)

## 📝 Notas Técnicas

### Requisitos
- Mínimo 2 snapshots (recomendado: 10+ para resultados significativos)
- Snapshots habilitados (`enable_snapshots = true`)
- Tiempo de cálculo: ~1-5 segundos para 50 snapshots (depende de hardware)

### Optimizaciones
- El análisis se ejecuta en un thread separado para no bloquear la simulación
- Se puede cancelar en cualquier momento
- Los resultados se cachean para evitar recálculos

## 🔗 Referencias

- `src/analysis/analysis.py`: Implementación del análisis
- `src/pipelines/pipeline_server.py`: Handler `handle_analyze_universe_atlas`
- Frontend: Componente de visualización de grafos (pendiente de implementar)
- `docs/40_Experiments/VISUALIZATION_OPTIMIZATION_ANALYSIS.md`: Análisis de optimización

---

*Última actualización: 2024-12-XX*

