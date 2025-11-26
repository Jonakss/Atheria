## 2025-11-21 - Corrección Crítica: Filtrado de Propagación Z en Motor Nativo

### Contexto
El usuario reportó problemas de rendimiento ("se tranca", "sin fps") y advertencias sobre "número sospechoso de coordenadas activas" (13k vs 4k esperadas).

### Problema Identificado
El motor nativo (C++) es tridimensional y propaga partículas a vecinos en Z (`z=-1` y `z=1`) incluso si la simulación se visualiza en 2D (`z=0`).
- `get_active_coords` retornaba ~3x coordenadas (z=-1, 0, 1).
- `NativeEngineWrapper` procesaba todas, sobrescribiendo el estado denso 2D múltiples veces.
- Esto causaba overhead innecesario y advertencias de duplicados.

### Solución Implementada
**Archivo:** `src/engines/native_engine_wrapper.py`

**Cambios:**
1.  **Filtrado Z=0:** En `_update_dense_state_from_sparse`, se ignoran explícitamente las coordenadas con `coord.z != 0`.
2.  **Robustez de Inicialización:** Se redujo el umbral de detección de partículas (`1e-9`) y se agregó lógica de reintento para evitar fallbacks a ruido aleatorio.

### Resultado
- ✅ Coordenadas procesadas reducidas de ~13k a ~4k (solo slice Z=0).
- ✅ Eliminación de advertencias de "coordenadas sospechosas".
- ✅ Mejora de rendimiento en conversión de estado.

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
