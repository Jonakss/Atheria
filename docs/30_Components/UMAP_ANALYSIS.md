# Análisis de Espacio de Estados (UMAP)

## 📋 Resumen

Visualización en tiempo real de la trayectoria del sistema en el espacio de fases utilizando **UMAP** (Uniform Manifold Approximation and Projection). A diferencia del "Atlas del Universo" (t-SNE sobre snapshots), este componente procesa un flujo continuo de estados para mostrar la dinámica en vivo.

## 🎯 Objetivo

Proporcionar una representación visual intuitiva de la complejidad dinámica del autómata celular cuántico. Permite identificar:
- **Atractores y Ciclos**: Trayectorias cerradas o puntos fijos.
- **Régimen Caótico**: Nubes dispersas de puntos.
- **Transiciones de Fase**: Movimientos bruscos entre clusters.

## 🔬 Metodología

### 1. Buffering Temporal
- `StateAnalyzer` mantiene un buffer circular de los últimos $N$ estados (por defecto 1000).
- Cada estado se aplana desde su dimensionalidad original (ej. $64 \times 64 \times 8 \approx 32k$ dimensiones).

### 2. Proyección UMAP
- Se ejecuta en un **hilo separado** (`daemon thread`) para no bloquear el bucle de simulación principal.
- Utiliza la biblioteca `umap-learn` para reducir la dimensionalidad de $\mathbb{R}^D$ a $\mathbb{R}^2$.
- **Métrica**: Distancia Euclidiana.
- **Vecinos**: 15 (balance local/global).
- **Distancia Mínima**: 0.1.

### 3. Integración en Flujo de Datos
- `DataProcessingService` alimenta el analizador con copias ligeras de los estados (`psi.cpu().numpy()`).
- Los resultados de la proyección (`x`, `y`) se adjuntan al payload del WebSocket `simulation_frame` bajo la clave `analysis_data`.

## 🛠️ Implementación

### Backend
- **Archivo**: `src/analysis/dimensionality.py`
- **Clase**: `StateAnalyzer`
- **Integración**: `src/services/data_processing_service.py`

```python
# Ejemplo de uso en servicio
self.state_analyzer.add_state(psi, step=step)
analysis_data = self.state_analyzer.get_latest_data()
```

### Frontend
- **Componente**: `frontend/src/components/analysis/AnalysisPanel.tsx`
- **Tecnología**: HTML5 Canvas (para renderizado eficiente de miles de puntos).
- **Ubicación UI**: Panel lateral izquierdo, pestaña "Analysis".

## 📊 Interpretación

- **Puntos**: Representan estados en el tiempo $t$.
- **Color/Brillo**: Indica la recencia del estado (puntos más brillantes son más recientes).
- **Líneas**: Conectan estados consecutivos, mostrando la trayectoria.
- **Clusters**: Indican regímenes dinámicos estables o metaestables.

## 🔄 Diferencias con Universe Atlas (t-SNE)

| Característica | UMAP (Este componente) | t-SNE (Atlas) |
|---|---|---|
| **Tiempo** | Real-time (stream) | Post-proceso (snapshots) |
| **Velocidad** | Rápido, incremental | Lento, global |
| **Objetivo** | Dinámica inmediata | Estructura global histórica |
| **Ejecución** | Thread background continuo | Job bajo demanda |

## 🔗 Dependencias
- `umap-learn`
- `scikit-learn` (opcional, para escalado)
- `numpy`, `torch`
