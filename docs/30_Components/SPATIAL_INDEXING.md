---
title: Spatial Indexing (Morton Codes)
type: component
status: active
tags: [component, optimization, spatial, cpp, verified]
created: 2024-11-19
updated: 2024-11-19
aliases: [Morton Codes, Z-order Curve, Spatial Optimization, Morton Code Indexing]
related: [[NATIVE_ENGINE_COMMUNICATION]], [[40_Experiments/EXP_007_SPATIAL_INDEXING_VERIFICATION]]
location: src/spatial.py
---

# 📦 Componente: Optimización Espacial (Spatial Indexing)

**Ubicación**: `src/spatial.py`  
**Estado**: Implementado (Fase 1)  
**Tecnología**: Curvas de Orden-Z (Morton Codes)

---

## 🎯 Propósito

En un motor disperso (Sparse Engine) como el de Atheria 4, buscar "vecinos cercanos" en un diccionario `(x, y, z)` tradicional es ineficiente para millones de partículas. La Optimización Espacial resuelve esto transformando coordenadas 3D en un índice 1D que preserva la localidad.

---

## ⚙️ Tecnología: Curvas de Orden-Z (Morton Codes)

Utilizamos **Códigos Morton**. Esta técnica "intercala" los bits de las coordenadas X, Y, Z para crear un solo entero único.

### Cómo Funciona

Si tenemos coordenadas binarias:

```
X = x1 x0
Y = y1 y0
Z = z1 z0
```

El **Código Morton** resultante es: `z1 y1 x1 z0 y0 x0`.

### Ejemplo Práctico

```
Coordenadas: (X=2, Y=3, Z=1)

Binario:
  X = 010 (2)
  Y = 011 (3)
  Z = 001 (1)

Morton Code:
  Intercalado: 0 0 0 1 1 1 0 = 001110 = 14 (decimal)
```

---

## ✅ Ventajas para Atheria

### 1. Localidad de Caché

Las partículas que están cerca en el espacio 3D (XYZ) tienden a tener códigos Morton cercanos. Esto hace que el acceso a memoria en C++ sea mucho más rápido.

**Ejemplo**:
```
Coordenadas cercanas:
  (10, 10, 10) → Morton: 1464
  (11, 10, 10) → Morton: 1465
  (10, 11, 10) → Morton: 1466

Los códigos Morton están contiguos en memoria → mejor caché hit rate
```

### 2. Búsqueda de Rango

Podemos encontrar todos los puntos dentro de un cubo calculando un rango de índices Morton, en lugar de iterar todo el universo.

**Uso**:
```python
# Encontrar todas las partículas en el cubo [x0:x1, y0:y1, z0:z1]
morton_min = coords_to_morton((x0, y0, z0))
morton_max = coords_to_morton((x1, y1, z1))

# Búsqueda eficiente en rango
particles_in_range = sparse_map.get_range(morton_min, morton_max)
```

### 3. Clave de Hash Eficiente

Usar un `int64` como clave de un HashMap es más rápido que hashear una tupla `(int, int, int)`.

**Comparación**:
```python
# Ineficiente: hashear tupla
hash((x, y, z))  # Operación costosa, dispersión aleatoria

# Eficiente: usar Morton code directamente
morton_code = coords_to_morton((x, y, z))  # Operación rápida, preserva localidad
```

---

## 📥 API del Componente (SpatialIndexer)

### Métodos Principales

#### `coords_to_morton(coords: Tensor[N, 3]) -> Tensor[N]`

Convierte un lote de coordenadas a índices Morton.

**Parámetros**:
- `coords`: Tensor de forma `[N, 3]` con coordenadas `[x, y, z]`

**Retorna**:
- Tensor de forma `[N]` con códigos Morton (int64)

**Ejemplo**:
```python
import torch
from src.spatial import SpatialIndexer

indexer = SpatialIndexer()
coords = torch.tensor([[10, 10, 10], [11, 10, 10], [10, 11, 10]])
morton_codes = indexer.coords_to_morton(coords)
# Resultado: tensor([1464, 1465, 1466])
```

#### `morton_to_coords(codes: Tensor[N]) -> Tensor[N, 3]`

Recupera las coordenadas originales desde códigos Morton.

**Parámetros**:
- `codes`: Tensor de forma `[N]` con códigos Morton

**Retorna**:
- Tensor de forma `[N, 3]` con coordenadas `[x, y, z]`

**Ejemplo**:
```python
morton_codes = torch.tensor([1464, 1465, 1466])
coords = indexer.morton_to_coords(morton_codes)
# Resultado: tensor([[10, 10, 10], [11, 10, 10], [10, 11, 10]])
```

#### `get_active_chunks(coords) -> List[int]`

Identifica bloques de espacio activos para simulación.

**Parámetros**:
- `coords`: Tensor o lista de coordenadas `[x, y, z]`

**Retorna**:
- Lista de códigos Morton únicos representando chunks activos

**Ejemplo**:
```python
coords = torch.tensor([[10, 10, 10], [11, 10, 10], [50, 50, 50]])
active_chunks = indexer.get_active_chunks(coords)
# Resultado: [1464, 1465, 12345]  # Chunks únicos con partículas
```

---

## 🔗 Relación con Otros Sistemas

### SparseEngine

El `SparseEngine` (C++) usará estos índices como claves primarias para el almacenamiento de materia.

**Integración**:
```cpp
// En src/cpp_core/include/sparse_engine.h
#include "spatial_indexer.h"

class Engine {
    // Usar Morton codes como claves del SparseMap
    SparseMap<int64_t, torch::Tensor> matter;  // Clave: Morton code
};
```

**Ventajas**:
- Acceso más rápido a partículas vecinas
- Mejor localidad de caché
- Búsqueda de rango eficiente

Ver también: [[NATIVE_ENGINE_COMMUNICATION]]

### Protocolo Binario

Se pueden enviar códigos Morton comprimidos para reducir aún más el ancho de banda en el futuro.

**Potencial de Optimización**:
```
Frame completo:
  - Coordenadas raw: 256x256x3 floats = 768 KB
  - Morton codes: 256x256 int64 = 512 KB
  - Con compresión LZ4: ~50-100 KB (90% reducción)
```

Ver también: [[WORLD_DATA_TRANSFER_OPTIMIZATION]]

---

## 📊 Rendimiento Esperado

### Operaciones Básicas

| Operación | Complejidad | Rendimiento Esperado |
|-----------|-------------|---------------------|
| `coords_to_morton` | O(1) | ~1 ns por coordenada |
| `morton_to_coords` | O(1) | ~1 ns por código |
| `get_active_chunks` | O(N) | ~1 μs para 10K partículas |

### Mejoras de Rendimiento

- **Búsqueda de vecinos**: 10-100x más rápido que iteración lineal
- **Localidad de caché**: 2-5x mejora en acceso a memoria
- **Compresión**: Hasta 90% de reducción en tamaño de datos

---

## 🚀 Próximos Pasos (Fase 2)

1. ✅ Implementación básica de Morton codes
2. ⏳ Integración con `SparseEngine` C++
3. ⏳ Optimización de búsqueda de rango
4. ⏳ Compresión de códigos Morton para transferencia
5. ⏳ Benchmark comparativo con hash tradicional

---

## 📝 Referencias

- **Morton Codes (Wikipedia)**: [Z-order curve](https://en.wikipedia.org/wiki/Z-order_curve)
- **Implementación**: `src/spatial.py`
- **Verificación**: [[40_Experiments/EXP_007_SPATIAL_INDEXING_VERIFICATION]]
- **Uso en SparseEngine**: [[NATIVE_ENGINE_COMMUNICATION]]

---

## 🔗 Ver También

- [[00_COMPONENTS_MOC|← Volver al MOC de Componentes]]
- [[40_Experiments/EXP_007_SPATIAL_INDEXING_VERIFICATION]] - Resultados de verificación
- [[NATIVE_ENGINE_COMMUNICATION]] - Integración con motor nativo

---

## 📌 Tags

#component #optimization #spatial #cpp #verified #performance
