# Sistema de Historia de Simulación

## ¿Cómo Funciona?

El sistema de historia permite guardar y analizar la evolución temporal de la simulación. Funciona en dos modos:

### 1. **Historia en Tiempo Real** (Automático)

Cuando la simulación está corriendo, puedes habilitar el guardado automático de frames:

- **Habilitar**: En `AdvancedControls`, activa "Habilitar Historia"
- **Intervalo**: Configura cada cuántos pasos se guarda un frame (por defecto: cada 10 pasos)
- **Límite**: Se mantienen los últimos N frames en memoria (por defecto: 1000)

**¿Qué se guarda?**
- `step`: Número de paso de la simulación
- `timestamp`: Momento en que se capturó el frame
- `map_data`: Datos del mapa (densidad, fase, etc.) según la visualización activa
- `hist_data`: Histogramas y estadísticas del frame

**Ubicación en memoria**: `g_state['simulation_history']` (objeto `SimulationHistory`)

### 2. **Historia desde Archivo** (Persistente)

Puedes guardar y cargar historiales completos:

- **Guardar**: En `AdvancedControls`, click en "Guardar Historia"
  - Se guarda en `output/simulation_history/simulation_history_YYYYMMDD_HHMMSS.json`
  - Incluye todos los frames acumulados en memoria
  
- **Cargar**: En `HistoryViewer` (tab "Historia"):
  - Lista todos los archivos guardados
  - Selecciona uno y click en "Cargar"
  - Los frames se cargan en memoria y puedes reproducirlos

**Formato del archivo**:
```json
{
  "metadata": {
    "total_frames": 100,
    "created_at": "2025-01-17T...",
    "max_frames": 1000
  },
  "frames": [
    {
      "step": 0,
      "timestamp": "...",
      "map_data": [[...], [...]],
      "hist_data": {...}
    },
    ...
  ]
}
```

## Visualización 3D de Evolución Temporal

El componente `History3DViewer` muestra la evolución apilando frames en el eje Z:

- **Eje X (rojo)**: Posición espacial X
- **Eje Y (verde)**: Valor/Altura del estado cuántico
- **Eje Z (azul)**: Tiempo (cada slice es un frame)

**Características**:
- Cada slice temporal es un plano transparente
- La altura (Y) representa el valor del estado en ese punto
- El color cambia con el tiempo (azul → verde → amarillo)
- Puedes ajustar la opacidad para ver mejor las capas internas
- Rotación automática para ver desde todos los ángulos

**Uso**:
1. Inicia la simulación
2. Los frames se acumulan automáticamente en tiempo real
3. O carga un historial guardado desde el tab "Historia"
4. El visualizador 3D muestra todos los frames apilados

## Comandos del Backend

```python
# Habilitar/deshabilitar historia
handle_enable_history(enabled: bool)

# Guardar historia actual
handle_save_history(filename: str = None) -> filepath

# Limpiar historia en memoria
handle_clear_history()

# Listar archivos guardados
handle_list_history_files() -> List[HistoryFile]

# Cargar archivo
handle_load_history_file(filename: str) -> frames
```

## Optimizaciones

- **Downsampling**: Los frames se guardan con el mismo tamaño que la visualización activa
- **Límite de memoria**: Solo se mantienen los últimos N frames (evita overflow)
- **Serialización eficiente**: Solo se guardan `map_data` y `hist_data` (no `flow_data`, `poincare_coords`, etc.)

## Casos de Uso

1. **Análisis de Patrones Temporales**: Ver cómo evolucionan las estructuras (gliders, ondas)
2. **Debugging**: Reproducir un comportamiento específico paso a paso
3. **Comparación**: Cargar diferentes historiales y comparar evoluciones
4. **Visualización 3D**: Ver la "película" completa de la simulación en 3D

