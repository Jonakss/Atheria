# EXP_006: Optimización de Transferencia de Datos

**Fecha**: 2024-11-19  
**Objetivo**: Experimentar con optimizaciones de transferencia de datos para reducir el ancho de banda y mejorar la latencia de visualización

## Contexto

El sistema actual envía datos de simulación vía WebSocket usando JSON, lo cual tiene varios problemas:

1. **Overhead de Base64**: ~33% de aumento innecesario
2. **JSON Verboso**: Formato ineficiente para datos numéricos
3. **Compresión Lenta**: zlib es relativamente lento
4. **Envío Completo**: Siempre envía el frame completo, incluso si solo cambió una parte

## Implementación de Optimizaciones

### 1. Binary WebSocket Frames

**Cambio**: Enviar datos binarios directamente en lugar de JSON + base64.

**Resultado Esperado**: ~33% de reducción en tamaño.

**Estado**: ✅ Implementado en `src/server/data_transfer_optimized.py`

### 2. LZ4 Compression

**Cambio**: Usar LZ4 en lugar de zlib para compresión.

**Resultado Esperado**: 2-3x más rápido que zlib con similar ratio.

**Estado**: ✅ Implementado (opcional, requiere `pip install lz4`)

### 3. Quantization (float32 → uint8)

**Cambio**: Cuantizar datos de visualización a uint8 (256 niveles).

**Resultado Esperado**: 4x reducción en tamaño (float32 = 4 bytes, uint8 = 1 byte).

**Estado**: ✅ Implementado (opcional, `use_quantization=True`)

### 4. Differential Compression

**Cambio**: Solo enviar diferencias entre frames consecutivos.

**Resultado Esperado**: 50-90% de ahorro cuando los cambios son pequeños.

**Estado**: ✅ Implementado (opcional, `use_differential=True`)

### 5. CBOR para Metadata

**Cambio**: Usar CBOR en lugar de JSON para metadata.

**Resultado Esperado**: ~20-30% más eficiente que JSON.

**Estado**: ✅ Implementado (opcional, requiere `pip install cbor2`)

## Métricas de Rendimiento

### Escenario de Prueba

- **Grid Size**: 256x256
- **d_state**: 11
- **FPS Objetivo**: 10 FPS
- **Visualización**: Density map

### Resultados Esperados

| Método | Tamaño por Frame | Ancho de Banda (10 FPS) | Velocidad Compresión |
|--------|------------------|-------------------------|---------------------|
| **JSON actual** | 50-100 KB | 500-1000 KB/s | Media |
| **Binary + LZ4** | 15-25 KB | 150-250 KB/s | Alta |
| **+ Quantization** | 5-10 KB | 50-100 KB/s | Alta |
| **+ Differential** | 1-5 KB | 10-50 KB/s | Alta |

**Mejora Total Esperada**: 5-10x reducción en ancho de banda, 2-3x más rápido.

## Scripts de Prueba

### Test de Transferencia

```bash
python tests/test_transfer_optimization.py
```

Este script compara:
- JSON vs Binary
- zlib vs LZ4
- Con/Sin quantization
- Con/Sin differential compression

## Uso en Producción

### Backend

Las optimizaciones están disponibles pero **no activadas por defecto** para mantener compatibilidad.

Para activar:

```python
# En src/pipelines/pipeline_server.py

# Activar optimizaciones
g_state['data_compression_enabled'] = True
g_state['use_binary_frames'] = True  # Si está implementado
g_state['use_quantization'] = True   # Opcional
g_state['use_differential'] = False  # Requiere estado previo
```

### Frontend

El frontend debe estar preparado para recibir frames binarios si se activa `use_binary_frames`.

**Estado**: ⚠️ Requiere implementación en frontend para frames binarios.

## Dependencias Opcionales

```bash
pip install lz4 cbor2
```

O usar el script:

```bash
./scripts/install_optimization_deps.sh
```

## Resultados Reales

> **Nota**: Los resultados reales se documentarán después de ejecutar los tests.

### Próximos Pasos

1. ✅ Implementación de optimizaciones
2. ⏳ Ejecutar benchmark completo
3. ⏳ Documentar resultados reales
4. ⏳ Activar en producción si los resultados son favorables
5. ⏳ Implementar soporte de frames binarios en frontend

## Referencias

- **Componente**: [[30_Components/WORLD_DATA_TRANSFER_OPTIMIZATION]] - Documentación técnica completa
- **Quick Start**: [[30_Components/QUICK_START_OPTIMIZATION]] - Guía rápida
- **Benchmark Script**: `tests/test_transfer_optimization.py`

## Tags

#experiment #optimization #performance #data-transfer #benchmark

