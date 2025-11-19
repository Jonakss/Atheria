# Zoom y Transferencia de Datos

## Estado Actual

### ¿Cómo Funciona el Zoom?

El zoom actualmente es **solo visual (CSS transform)** en el frontend. Esto significa:

1. **Se envía TODO el grid completo** desde el backend siempre
2. El zoom solo escala visualmente el canvas usando `transform: scale()`
3. **No se reduce la cantidad de datos** enviados cuando haces zoom

### Ejemplo

Si tienes un grid de 256x256:
- **Datos enviados:** 256 × 256 = 65,536 valores (siempre)
- **Zoom 2x:** Visualmente se ve más grande, pero se siguen enviando 65,536 valores
- **Zoom 0.5x:** Visualmente se ve más pequeño, pero se siguen enviando 65,536 valores

### Ventajas del Enfoque Actual

✅ **Simplicidad:** No requiere lógica compleja de region-of-interest  
✅ **Calidad:** Siempre tienes todos los datos para análisis  
✅ **Sin latencia adicional:** No hay cálculos extra de cropping/downsampling

### Desventajas

❌ **Ancho de banda:** Se envía más datos de los necesarios cuando haces zoom out  
❌ **Memoria:** Se procesan todos los datos aunque no se vean  
❌ **Rendimiento:** Puede ser lento en grids muy grandes (512x512+)

---

## Optimización: Downsampling Opcional

### Implementación

Se ha añadido un parámetro opcional `downsample_factor` a `get_visualization_data()`:

```python
def get_visualization_data(psi, viz_type, delta_psi=None, motor=None, downsample_factor=1):
    """
    downsample_factor: Factor de reducción (1 = sin reducción, 2 = mitad, 4 = cuarto)
    """
    if downsample_factor > 1:
        # Reducir resolución usando promedio
        # ...
```

### Cómo Usar

**Opción 1: Downsampling Fijo**
```python
# En pipeline_server.py
viz_data = get_visualization_data(
    psi, 
    viz_type,
    delta_psi=delta_psi,
    motor=motor,
    downsample_factor=2  # Reducir a la mitad
)
```

**Opción 2: Downsampling Basado en Zoom (Futuro)**
```python
# Calcular factor basado en zoom del cliente
zoom_level = g_state.get('client_zoom', 1.0)
if zoom_level < 0.5:  # Zoom out
    downsample_factor = 2  # Reducir datos
else:
    downsample_factor = 1  # Datos completos
```

### Trade-offs

| Factor | Reducción | Datos Enviados | Calidad Visual | Rendimiento |
|--------|-----------|----------------|----------------|-------------|
| 1      | 0%        | 100%           | Máxima         | Normal      |
| 2      | 50%       | 25%            | Buena          | +4x más rápido |
| 4      | 75%       | 6.25%          | Aceptable      | +16x más rápido |

---

## Recomendaciones

### Para Grids Pequeños (< 128x128)
- ✅ **No usar downsampling** - El overhead no vale la pena
- ✅ Zoom CSS es suficiente

### Para Grids Medianos (128-256)
- ⚠️ **Downsampling opcional** - Solo si hay problemas de rendimiento
- ✅ Zoom CSS funciona bien

### Para Grids Grandes (256+)
- ✅ **Considerar downsampling** - Especialmente si hay lag
- ⚠️ **Downsampling adaptativo** - Basado en zoom level

---

## Implementación Futura: Region of Interest (ROI)

### Concepto

En lugar de enviar todo el grid, enviar solo la región visible (viewport):

```
Grid completo: 512x512
Viewport visible (zoom 2x): 256x256
Datos a enviar: Solo 256x256 (75% menos)
```

### Ventajas

✅ **Reducción masiva de datos** cuando haces zoom in  
✅ **Mejor rendimiento** en grids grandes  
✅ **Menor latencia** de red

### Desventajas

❌ **Complejidad:** Requiere tracking de viewport en frontend  
❌ **Latencia adicional:** Cálculo de ROI en backend  
❌ **Problemas con pan:** Necesita actualizar ROI constantemente

### Implementación Propuesta

```python
# Frontend envía viewport info
viewport = {
    'x': pan.x,
    'y': pan.y,
    'width': canvas.width / zoom,
    'height': canvas.height / zoom
}

# Backend calcula ROI
roi_data = get_visualization_data(
    psi,
    viz_type,
    viewport=viewport,  # Nueva opción
    downsample_factor=1
)
```

---

## Estado Actual del Código

### ✅ Implementado

1. **Downsampling opcional** en `pipeline_viz.py`
   - Parámetro `downsample_factor`
   - Usa promedio para reducir resolución

2. **Zoom CSS** en `PanZoomCanvas.tsx`
   - Transform scale para zoom visual
   - Throttling para mejor rendimiento

### ⏳ Pendiente

1. **Downsampling adaptativo** basado en zoom
2. **Region of Interest (ROI)** para zoom in
3. **Métricas de ancho de banda** para monitorear transferencia

---

## Cómo Verificar Datos Enviados

### En el Frontend

Abre DevTools → Network → WS (WebSocket):
- Busca mensajes `simulation_frame`
- Revisa el tamaño del payload
- Compara con/sin downsampling

### En el Backend

Añade logging:
```python
import sys
map_data_size = sys.getsizeof(str(frame_payload['map_data']))
logging.info(f"Datos enviados: {map_data_size / 1024:.2f} KB")
```

---

## Conclusión

**Actualmente:** El zoom es solo visual, se envían todos los datos siempre.

**Optimización disponible:** Downsampling opcional (no activado por defecto).

**Futuro:** ROI adaptativo para grids grandes.

**Recomendación:** Para la mayoría de casos, el zoom CSS actual es suficiente. Solo activar downsampling si hay problemas de rendimiento con grids grandes (>256x256).

