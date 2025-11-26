## 2025-11-20 - Separación Live Feed: Binario (MessagePack) vs JSON

### Contexto
Los datos de visualización (live feed) son muy grandes (arrays numéricos de 256x256) y enviarlos como JSON es ineficiente. Se decidió separar:
- **JSON**: Solo para comandos, notificaciones y metadatos del servidor (pequeños)
- **Binario (MessagePack/CBOR)**: Para frames de visualización (grandes, arrays numéricos)

### Implementación

#### Backend (`src/server/data_serialization.py`):
- `serialize_frame_binary()`: Serializa frames de visualización a binario (MessagePack → CBOR → JSON fallback)
- `deserialize_frame_binary()`: Deserializa frames binarios
- `should_use_binary()`: Determina si un mensaje debe usar binario o JSON

#### Backend (`src/server/server_state.py`):
- `broadcast()` actualizado: Detecta automáticamente si es `simulation_frame` y usa binario
- Estrategia híbrida: Envía metadata JSON primero (~100 bytes), luego datos binarios
- Logging detallado del formato usado y tamaño

#### Frontend (`frontend/src/utils/dataDecompression.ts`):
- `decodeBinaryFrame()` actualizado: Soporta MessagePack, CBOR y JSON
- Auto-detección de formato por primer byte
- Soporte para formato especificado desde metadata

#### Frontend (`frontend/src/context/WebSocketContext.tsx`):
- Manejo de mensajes híbridos: Detecta metadata JSON seguida de datos binarios
- `pendingBinaryFormat` ref: Almacena formato esperado entre mensajes
- Procesamiento correcto de frames binarios con metadata separada

### Beneficios
- **Reducción de tamaño**: MessagePack es 3-5x más compacto que JSON para arrays numéricos
- **Mejor rendimiento**: Menos parsing, menos transferencia de datos
- **Separación clara**: JSON solo para comandos/metadatos, binario para datos grandes
- **Retrocompatibilidad**: Fallback a JSON si MessagePack/CBOR no está disponible

### Formato de Mensaje Híbrido
1. **Metadata JSON** (pequeño, ~100 bytes):
   ```json
   {
     "type": "simulation_frame_binary",
     "format": "msgpack",
     "size": 15234
   }
   ```
2. **Datos Binarios** (grande, MessagePack/CBOR serializado)

### Referencias
- `src/server/data_serialization.py` - Serialización binaria eficiente
- `src/server/server_state.py` - Función `broadcast()` actualizada
- `frontend/src/utils/dataDecompression.ts` - Decodificación binaria
- `frontend/src/context/WebSocketContext.tsx` - Manejo de mensajes híbridos

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
