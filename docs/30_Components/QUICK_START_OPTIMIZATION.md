# 🚀 Quick Start: Optimización de Transferencia de Datos

## Instalación Rápida

```bash
# Instalar dependencias opcionales
pip install lz4 cbor2

# O usar el script
./scripts/install_optimization_deps.sh
```

## Comparación Rápida

| Método | Tamaño (256x256) | Velocidad | Ancho de Banda (10 FPS) |
|--------|------------------|-----------|-------------------------|
| **JSON actual** | 50-100 KB | Media | 500-1000 KB/s |
| **Binary optimizado** | 10-20 KB | Alta | 100-200 KB/s |
| **Mejora** | **5-10x menor** | **2-3x más rápido** | **5-10x ahorro** |

## Uso en Backend

```python
# En src/pipeline_server.py, en simulation_loop:

from .data_transfer_optimized import optimize_frame_payload_binary
from .server_state import broadcast_binary

# En lugar de:
# frame_payload = await optimize_frame_payload(...)
# await broadcast({"type": "simulation_frame", "payload": frame_payload})

# Usar:
frame_payload_binary = await optimize_frame_payload_binary(
    frame_payload_raw,
    use_quantization=True,      # float32→uint8 (4x reducción)
    use_differential=False,      # Opcional: solo cambios
    previous_frame=g_state.get('last_frame')
)

await broadcast_binary(frame_payload_binary, frame_type="simulation_frame")

# Guardar para differential compression (opcional)
g_state['last_frame'] = frame_payload_raw
```

## Uso en Frontend

```typescript
// En frontend/src/context/WebSocketContext.tsx

// Detectar mensaje binario
socket.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
        // Frame binario optimizado
        const arrayBuffer = event.data instanceof Blob 
            ? await event.data.arrayBuffer() 
            : event.data;
        
        const payload = decodeFrameBinary(new Uint8Array(arrayBuffer));
        handleSimulationFrame(payload);
    } else {
        // Mensaje JSON normal
        const message = JSON.parse(event.data);
        handleMessage(message);
    }
};

// Función de decodificación (necesitarás implementar decodeFrameBinary)
// Ver src/data_transfer_optimized.py para el formato
```

## Verificar Mejoras

```bash
# Ejecutar benchmark comparativo
python3 tests/test_transfer_optimization.py
```

Esto mostrará:
- Tamaño reducido (esperado: 5-10x)
- Velocidad mejorada (esperado: 2-3x)
- Ahorro de ancho de banda (esperado: 5-10x)

## Activación Gradual

Puedes activar las optimizaciones gradualmente:

1. **Fase 1**: Instalar dependencias (sin cambios de código)
2. **Fase 2**: Habilitar binary frames solo para frames grandes (>100 KB)
3. **Fase 3**: Habilitar quantization (sin differential)
4. **Fase 4**: Habilitar differential compression (experimental)

## Troubleshooting

**Si las dependencias no están disponibles**:
- El sistema usará fallbacks automáticos (zlib en lugar de LZ4, JSON en lugar de CBOR)
- Funcionará pero sin las mejoras de rendimiento

**Si el frontend no recibe datos**:
- Verificar que el WebSocket soporta binary frames
- Usar fallback base64 (menos eficiente pero compatible)

**Ver logs**:
```bash
# Buscar en logs del servidor:
grep "Payload size" logs/*.log
```

## Beneficios Esperados

✅ **5-10x reducción** de ancho de banda  
✅ **2-3x más rápido** en compresión  
✅ **Menor latencia** (serialización más rápida)  
✅ **Menor CPU** en backend y frontend  
✅ **Mejor experiencia** para usuarios con conexiones lentas  

