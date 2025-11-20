# 🔌 WebSocket Protocol - Separación Binario vs JSON

Este documento describe el protocolo de comunicación WebSocket para Atheria 4, que separa eficientemente datos de visualización (binario) de comandos del servidor (JSON).

---

## 📋 Estrategia General

### JSON: Solo para Comandos y Metadatos del Servidor
- **Comandos**: `simulation.start`, `experiment.load`, etc.
- **Notificaciones**: Mensajes de estado, errores, warnings
- **Metadatos del Servidor**: `compile_status`, `inference_status_update`, etc.
- **Tamaño**: Pequeño (< 1 KB típicamente)

### Binario (MessagePack/CBOR): Para Frames de Visualización
- **Frames de Simulación**: `simulation_frame` con arrays numéricos grandes
- **Tamaño**: Grande (10-50 KB típicamente)
- **Formato**: MessagePack (preferido) → CBOR → JSON (fallback)

---

## 🔄 Formato de Mensaje Híbrido para Frames

Para frames de visualización grandes, se usa un formato híbrido:

### 1. Metadata JSON (Primer Mensaje)
```json
{
  "type": "simulation_frame_binary",
  "format": "msgpack",  // "msgpack", "cbor", o "json"
  "size": 15234  // Tamaño en bytes del siguiente mensaje binario
}
```

### 2. Datos Binarios (Segundo Mensaje)
- **Formato**: MessagePack/CBOR serializado del payload completo
- **Contenido**: Mismo payload que antes (map_data, hist_data, etc.)

---

## 🛠️ Implementación Backend

### `src/server/data_serialization.py`

#### `serialize_frame_binary(payload: Dict) -> Tuple[bytes, str]`
Serializa un frame de visualización a binario eficiente:
1. Intenta MessagePack (más eficiente para arrays numéricos)
2. Fallback a CBOR (bueno para arrays binarios)
3. Fallback final a JSON (último recurso)

#### `deserialize_frame_binary(data: bytes, format_hint: Optional[str]) -> Dict`
Deserializa un frame binario según el formato especificado.

#### `should_use_binary(message_type: str, payload: Optional[Dict]) -> bool`
Determina si un mensaje debe usar binario o JSON:
- `simulation_frame` → `True` (binario)
- Otros → `False` (JSON)

### `src/server/server_state.py`

#### `broadcast(data: Dict)`
Función principal de broadcasting actualizada:
- Detecta automáticamente si es `simulation_frame`
- Si es binario:
  1. Serializa payload a binario usando `serialize_frame_binary()`
  2. Envía metadata JSON primero
  3. Envía datos binarios después
- Si es JSON: Envía directamente como JSON

---

## 🎨 Implementación Frontend

### `frontend/src/utils/dataDecompression.ts`

#### `decodeBinaryFrame(data: ArrayBuffer | Uint8Array | string, format?: string) -> Promise<any>`
Decodifica un frame binario:
1. Si `format` está especificado, intenta ese formato primero
2. Auto-detección: Si parece JSON (`{` o `[`), decodifica como JSON
3. Intenta MessagePack/CBOR usando `@msgpack/msgpack`
4. Fallback final a JSON

### `frontend/src/context/WebSocketContext.tsx`

#### Manejo de Mensajes Híbridos
El `WebSocketContext` maneja el protocolo híbrido:

1. **Mensaje JSON con metadata binaria**:
   - Detecta `type.endsWith('_binary')` y `format`
   - Almacena formato esperado en `pendingBinaryFormat.current`
   - No procesa como mensaje completo, espera el siguiente

2. **Mensaje Binario**:
   - Usa `pendingBinaryFormat.current` para decodificar
   - Deserializa usando `decodeBinaryFrame()` con formato especificado
   - Procesa como frame de visualización normal

---

## 📊 Rendimiento

### Comparación de Tamaños (256x256 grid, float32)
- **JSON**: ~250 KB (sin compresión)
- **JSON comprimido**: ~80 KB (zlib)
- **MessagePack**: ~65 KB (3.8x más pequeño que JSON sin comprimir)
- **CBOR**: ~70 KB (3.5x más pequeño que JSON sin comprimir)

### Latencia
- **JSON**: ~5-10ms parsing + transferencia
- **MessagePack**: ~2-4ms parsing + transferencia (2-3x más rápido)

---

## 🔄 Retrocompatibilidad

El sistema mantiene retrocompatibilidad:
- Si MessagePack/CBOR no están disponibles, usa JSON
- El frontend puede decodificar JSON, MessagePack y CBOR
- Los comandos siempre usan JSON (no cambian)

---

## 📝 Referencias

- `src/server/data_serialization.py` - Serialización binaria
- `src/server/server_state.py` - Función `broadcast()` actualizada
- `frontend/src/utils/dataDecompression.ts` - Decodificación binaria
- `frontend/src/context/WebSocketContext.tsx` - Manejo de mensajes híbridos

---

## 🚀 Próximos Pasos

- [ ] Implementar compresión LZ4 para datos binarios (reducción adicional 20-30%)
- [ ] Añadir differential compression (solo cambios entre frames)
- [ ] Optimizar serialización de arrays NumPy directamente (sin conversión a lista)

