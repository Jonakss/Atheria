# Optimización de Transferencia de Datos del Mundo

## Problema Actual

El sistema actual envía datos del mundo (simulación) via WebSocket usando JSON, lo cual tiene varios problemas de eficiencia:

1. **Base64 Overhead**: Aumenta el tamaño en ~33% innecesariamente
2. **JSON Verbosidad**: JSON es muy verboso para datos numéricos
3. **Compresión Lenta**: zlib es relativamente lento
4. **Serialización Ineficiente**: Convertir arrays NumPy → lista → JSON es lento
5. **Envío Completo**: Envía el frame completo cada vez, incluso si solo cambió una pequeña parte

**Ejemplo de Tamaño**:
- Grid 256x256 con float32 = 256 KB sin compresión
- Con base64: ~341 KB (+33%)
- Con zlib + base64: ~50-100 KB (depende del contenido)
- **Objetivo**: < 20 KB con mejor compresión

## Mejoras Propuestas

### 1. **Binary WebSocket Frames** (Sin Base64)

**Mejora**: Enviar datos binarios directamente por WebSocket, eliminando el overhead de base64.

**Ahorro**: ~33% de reducción inmediata

**Implementación**:
```python
# Antes (JSON + base64)
await ws.send_json({"map_data": base64_string})  # ~341 KB

# Después (Binary)
await ws.send_bytes(binary_data)  # ~256 KB
```

### 2. **LZ4 Compression** (Más Rápido que zlib)

**Mejora**: LZ4 es 2-3x más rápido que zlib con similar ratio de compresión para datos numéricos.

**Ahorro**: ~30-50% de tiempo de compresión

**Instalación**:
```bash
pip install lz4
```

### 3. **Quantization** (float32 → uint8)

**Mejora**: Cuantizar datos de visualización a uint8 (256 valores) en lugar de float32. Para visualización, 256 niveles son más que suficientes.

**Ahorro**: 4x reducción de tamaño (256 KB → 64 KB)

**Proceso**:
```python
# Normalizar a [0, 1]
normalized = (arr - min) / (max - min)
# Cuantizar a uint8
quantized = (normalized * 255).astype(np.uint8)
# Descuantizar en frontend
denormalized = quantized / 255.0 * (max - min) + min
```

**Pérdida de Precisión**: Mínima para visualización (256 niveles es suficiente)

### 4. **Differential Compression** (Solo Cambios)

**Mejora**: En lugar de enviar el frame completo, enviar solo las diferencias desde el frame anterior.

**Ahorro**: 50-90% en simulaciones con cambios lentos

**Implementación**:
```python
if use_differential and previous_frame:
    diff = current_frame - previous_frame
    if np.abs(diff).max() < threshold:
        # Enviar solo diff (mucho más pequeño)
        send(diff)
    else:
        # Cambios grandes, enviar completo
        send(current_frame)
```

### 5. **CBOR** (Binary JSON)

**Mejora**: CBOR (Concise Binary Object Representation) es más eficiente que JSON para datos estructurados.

**Ahorro**: ~20% en metadata y estructuras

**Instalación**:
```bash
pip install cbor2
```

## Implementación Híbrida Recomendada

### Estrategia por Tipo de Dato

1. **Metadata** (step, timestamp, simulation_info):
   - CBOR (pequeño, eficiente)
   - ~100 bytes

2. **map_data** (grid principal):
   - Quantization (float32→uint8)
   - LZ4 compression
   - Binary format
   - ~10-20 KB (vs ~50-100 KB actual)

3. **complex_3d_data, flow_data, phase_hsv_data**:
   - Solo si se están usando (no enviar siempre)
   - Quantization + LZ4
   - Binary format

4. **Differential compression**:
   - Opcional (activar para simulaciones lentas)
   - Ahorra 50-90% adicional

### Tamaño Estimado Final

**Antes** (JSON + zlib + base64):
- Grid 256x256: ~50-100 KB por frame
- A 10 FPS: 500-1000 KB/s

**Después** (Binary + Quantization + LZ4):
- Grid 256x256: ~10-20 KB por frame
- A 10 FPS: 100-200 KB/s

**Mejora**: 5-10x reducción de ancho de banda

## Plan de Migración

### Fase 1: Instalación de Dependencias

```bash
pip install lz4 cbor2
```

### Fase 2: Backend - Actualizar `broadcast()` para usar Binary

```python
# src/server_state.py
async def broadcast_binary(data: bytes):
    """Envía datos binarios a todos los clientes WebSocket."""
    if not g_state['websockets']:
        return
    
    tasks = []
    for ws in list(g_state['websockets'].values()):
        if not ws.closed:
            tasks.append(ws.send_bytes(data))  # ← Nuevo método
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
```

### Fase 3: Backend - Usar Nueva Codificación

```python
# src/pipeline_server.py
from .data_transfer_optimized import optimize_frame_payload_binary

# En simulation_loop:
frame_payload_binary = await optimize_frame_payload_binary(
    frame_payload_raw,
    use_quantization=True,
    use_differential=True,  # Opcional
    previous_frame=g_state.get('last_frame')
)

# Enviar como binary
await broadcast_binary(frame_payload_binary)

# Guardar para differential
g_state['last_frame'] = frame_payload_raw
```

### Fase 4: Frontend - Decodificar Binary Frames

```typescript
// frontend/src/context/WebSocketContext.tsx
socket.onmessage = async (event) => {
    // Detectar si es binario o JSON
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
        // Frame binario optimizado
        const payload = await decodeFrameBinary(event.data);
        handleSimulationFrame(payload);
    } else {
        // Mensaje JSON normal (comandos, notificaciones, etc.)
        const message = JSON.parse(event.data);
        handleMessage(message);
    }
};
```

## Comparación de Rendimiento

### Métricas Esperadas

| Métrica | Actual (JSON) | Optimizado (Binary) | Mejora |
|---------|---------------|---------------------|--------|
| Tamaño por frame | 50-100 KB | 10-20 KB | 5-10x |
| Tiempo de compresión | 5-10 ms | 2-3 ms | 2-3x |
| Tiempo de serialización | 10-20 ms | 1-2 ms | 10x |
| Ancho de banda (10 FPS) | 500-1000 KB/s | 100-200 KB/s | 5-10x |
| CPU backend | Alta | Media | 50% |
| CPU frontend | Media | Baja | 50% |

### Benchmarks Realistas

Para un grid 256x256:
- **Actual**: ~80 KB/frame, ~800 KB/s a 10 FPS
- **Optimizado**: ~15 KB/frame, ~150 KB/s a 10 FPS
- **Mejora**: 5.3x reducción de ancho de banda

## Consideraciones

### Compatibilidad

- **Backward Compatibility**: Mantener soporte para JSON durante la transición
- **Feature Detection**: Frontend puede detectar si el servidor soporta binary frames
- **Fallback**: Si binary no está disponible, usar JSON optimizado

### Precisión

- **Quantization**: Solo afecta visualización, no afecta la simulación
- **256 niveles**: Más que suficiente para visualización humana
- **Pérdida imperceptible**: Diferencias no visibles a simple vista

### Latencia

- **Compresión más rápida**: LZ4 reduce latencia
- **Serialización más rápida**: Binary es más rápido que JSON
- **Resultado**: Latencia total reducida en ~50%

## Próximos Pasos

1. ✅ Crear `data_transfer_optimized.py` con implementación binaria
2. ⏳ Instalar dependencias (`lz4`, `cbor2`)
3. ⏳ Actualizar `broadcast()` para soportar binary frames
4. ⏳ Actualizar `simulation_loop()` para usar nueva codificación
5. ⏳ Actualizar frontend para decodificar binary frames
6. ⏳ Benchmark comparativo (actual vs optimizado)
7. ⏳ Documentar y hacer commit

## Referencias

- [LZ4 Compression](https://github.com/lz4/lz4)
- [CBOR Specification](https://cbor.io/)
- [WebSocket Binary Frames](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/send)
- [Quantization for Visualization](https://en.wikipedia.org/wiki/Quantization_(image_processing))

