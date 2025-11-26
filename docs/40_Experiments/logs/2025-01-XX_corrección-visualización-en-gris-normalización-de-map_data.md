## 2025-01-XX - Corrección: Visualización en Gris (Normalización de map_data)

### Problema
La visualización siempre cargaba en gris y no mostraba datos, incluso cuando había datos válidos.

### Causa Raíz
En `src/pipelines/viz/utils.py`, la función `normalize_map_data()` retornaba un array de ceros cuando todos los valores eran iguales (`max_val == min_val`), lo que causaba que la visualización apareciera completamente gris/negra.

### Solución Implementada

**1. Mejora de `normalize_map_data()`:**
- Si todos los valores son iguales, retorna `0.5` (gris medio) en lugar de ceros
- Permite ver que hay datos aunque no haya variación
- Usa `float32` para mejor rendimiento

**2. Validaciones Adicionales:**
- Verificación de `map_data` vacío antes de normalizar
- Fallback a densidad si está vacío
- Validación de forma (debe ser 2D)
- Reshape automático si la forma es incorrecta

**3. Logging para Debugging:**
- Advertencias cuando `map_data` tiene problemas
- Logs de rango de valores para diagnóstico

### Archivos Modificados
- `src/pipelines/viz/utils.py` - Función `normalize_map_data()` mejorada
- `src/pipelines/viz/core.py` - Validaciones adicionales antes de normalizar

### Resultado
- Visualización muestra gris medio cuando todos los valores son iguales
- Mejor manejo de casos edge (arrays vacíos, formas incorrectas)
- Logging útil para debugging

### Referencias
- `src/pipelines/viz/utils.py` - Normalización de map_data
- `src/pipelines/viz/core.py` - Validaciones de map_data

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
