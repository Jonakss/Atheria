# 2025-11-28: Implementación de OctreeIndex (Morton Codes)

## Objetivo
Implementar un índice espacial eficiente (Octree lineal) usando códigos Morton (Z-order curve) para acelerar las búsquedas espaciales en el motor nativo C++.

## Cambios Realizados

### 1. Utilidades de Morton (`morton_utils.h`)
- Se implementaron funciones para intercalar bits (bit-interleaving) de coordenadas 3D.
- Soporte para coordenadas de 21 bits por dimensión (rango [0, ~2M]).
- Funciones `coord_to_morton` y `morton_to_coord`.

### 2. Clase OctreeIndex (`octree.h`, `octree.cpp`)
- Implementación de un contenedor para códigos Morton ordenados.
- Métodos:
    - `insert(coord)`: Agrega una coordenada y marca el índice como "sucio".
    - `build()`: Ordena y elimina duplicados.
    - `clear()`: Limpia el índice.

### 3. Integración en Engine (`sparse_engine.cpp`)
- Se agregó `OctreeIndex octree_` como miembro de la clase `Engine`.
- Se actualizó `add_particle` para insertar automáticamente en el octree.
- Se actualizó `step_native` para asegurar que el octree esté construido (`build()`) antes de cada paso.

### 4. Verificación
- Se creó un script de prueba `tests/test_octree_manual.py`.
- Se verificó que las partículas agregadas al motor se reflejan correctamente en el conteo de activos después de un paso de simulación.
- El sistema maneja correctamente duplicados.

## Resultados
- El motor nativo ahora mantiene un índice espacial ordenado automáticamente.
- Esto sienta las bases para optimizaciones futuras como búsqueda de vecinos en $O(\log N)$ y detección de colisiones eficiente.

## Próximos Pasos
- Implementar métodos de búsqueda de rango (`query_radius`) en `OctreeIndex`.
- Usar el octree para optimizar la generación de vacío (`activate_neighborhood`).
