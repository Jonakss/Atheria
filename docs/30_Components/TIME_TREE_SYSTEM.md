# Sistema de Árbol de Tiempo (Time-Travel Debugging)

## Concepto

El sistema de "Árbol de Tiempo" permite almacenar la historia de simulación de forma eficiente usando **keyframes** y **deltas** en lugar de guardar frames completos. Esto permite navegación temporal instantánea con un costo de memoria mínimo.

## ¿Cómo Funciona?

### Sin Octrees (Implementación Actual)

En lugar de usar estructuras de datos complejas como Octrees, usamos una estrategia simple pero efectiva:

1. **Keyframes Completos**: Guardamos frames completos cada N frames (por defecto cada 10 frames)
2. **Deltas (Diferencias)**: Entre keyframes, guardamos solo las diferencias (deltas)
3. **Compresión Sparse**: Solo guardamos las posiciones donde hay cambios significativos (>1% del máximo)

### Ventajas

- **Memoria Eficiente**: En lugar de guardar 1000 frames completos, guardamos ~100 keyframes + deltas pequeños
- **Navegación Rápida**: Para reconstruir cualquier frame, solo necesitamos:
  - 1 keyframe (carga)
  - Aplicar deltas hasta el frame deseado (muy rápido)
- **Escalable**: Funciona bien hasta millones de frames
- **Simple**: No requiere estructuras de datos complejas

### Ejemplo

```
Frame 0:  [Keyframe completo] ──────────────┐
Frame 1:  [Delta desde Frame 0]              │
Frame 2:  [Delta desde Frame 0]              │
...                                           │
Frame 10: [Keyframe completo] ───────────────┤ Cada 10 frames
Frame 11: [Delta desde Frame 10]             │
Frame 12: [Delta desde Frame 10]             │
...
```

Para reconstruir Frame 7:
1. Cargar Frame 0 (keyframe)
2. Aplicar deltas de Frame 1-7
3. ¡Listo!

## Uso

### En el Backend

```python
from src.time_tree_manager import TimeTreeManager

# Crear manager para un experimento
tree = TimeTreeManager(
    experiment_name="mi_experimento",
    keyframe_interval=10,  # Keyframe cada 10 frames
    max_delta_size=0.1     # Deltas hasta 10% del tamaño original
)

# Agregar frames
for step in range(1000):
    frame_data = get_simulation_frame(step)  # numpy array
    tree.add_frame(step, frame_data)

# Reconstruir cualquier frame
frame_42 = tree.get_frame(42)  # Reconstruye desde keyframes + deltas

# Estadísticas
stats = tree.get_statistics()
print(f"Compresión: {stats['compression_ratio']:.2f}x")
print(f"Tamaño total: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
```

### Integración con Historia Actual

El sistema se puede usar junto con `SimulationHistory`:

- **SimulationHistory**: Para análisis rápido y visualización (últimos N frames)
- **TimeTreeManager**: Para almacenamiento a largo plazo y navegación temporal completa

## Configuración

### Parámetros Importantes

- **`keyframe_interval`**: 
  - Valores más pequeños = más precisión, más memoria
  - Valores más grandes = menos memoria, menos precisión
  - Recomendado: 10-50 frames

- **`max_delta_size`**:
  - Si un delta es > este valor, se guarda como keyframe completo
  - Recomendado: 0.1 (10% del tamaño original)

## Futuras Mejoras

1. **Compresión de Deltas**: Usar algoritmos de compresión (zlib, lz4) para deltas
2. **Jerarquía Temporal**: Agregar niveles de keyframes (cada 10, cada 100, cada 1000)
3. **Streaming**: Cargar keyframes y deltas bajo demanda
4. **Visualización Temporal**: UI para navegar por el árbol de tiempo
5. **Búsqueda Temporal**: Encontrar frames con características específicas

## Comparación con Octrees

| Característica | TimeTree (Actual) | Octrees (Futuro) |
|---------------|-------------------|------------------|
| Complejidad | Simple | Compleja |
| Memoria | Buena | Excelente |
| Navegación | Rápida | Instantánea |
| Escalabilidad | Millones | Billones |
| Implementación | ✅ Listo | 🔄 Futuro |

## Referencias

- Inspirado en el concepto de "Time-Travel Debugging" de sistemas de simulación
- Similar a técnicas de compresión de video (I-frames, P-frames, B-frames)
- Base para futura implementación de Octrees 4D (Espacio + Tiempo)

