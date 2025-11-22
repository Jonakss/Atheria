# Motor Sparse: Celdas Activas y Vecinos

## Concepto

El motor sparse (disperso) en C++ procesa **no solo las celdas activas**, sino también **sus vecinos** (incluso si están vacíos). Esto es crucial porque:

1. **Las celdas activas pueden modificar el vacío circundante**: Aunque una celda vecina esté vacía, las celdas activas pueden influir en ella durante el procesamiento.
2. **Analogía con QFT**: En teoría cuántica de campos, el vacío genera partículas virtuales y puede ser modificado por campos cercanos. En Atheria, el vacío se representa mediante `HarmonicVacuum`, que genera fluctuaciones deterministas.
3. **Propagación de efectos**: Los efectos de las celdas activas se propagan a sus vecinos, activando nuevas regiones.

## Implementación Actual

### Activación de Vecindarios

Cuando se agrega una partícula a una coordenada `(x, y)`, el motor **activa automáticamente su vecindario**:

```cpp
void Engine::add_particle(const Coord3D& coord, const torch::Tensor& state) {
    matter_map_.insert_tensor(coord, state_on_device);
    activate_neighborhood(coord);  // ← Activa vecinos
}
```

`activate_neighborhood()` (radio=1 por defecto) activa las 8 celdas vecinas en 2D:
- Norte, Sur, Este, Oeste
- Noreste, Noroeste, Sureste, Suroeste

### Procesamiento de Celdas Activas

En `step_native()`, el motor:

1. **Procesa todas las celdas en `active_region_`** (línea 97):
   ```cpp
   std::vector<Coord3D> processed_coords(active_region_.begin(), active_region_.end());
   ```

2. **Para cada celda activa, obtiene su estado y el de sus vecinos**:
   ```cpp
   torch::Tensor current_state = get_state_at(coord);  // Materia o vacío
   ```

3. **Construye un "patch" alrededor de cada celda** para el modelo:
   - Si la celda vecina tiene materia → usa el estado de la materia
   - Si la celda vecina está vacía → usa `vacuum_.get_fluctuation()` (vacío cuántico)

### Vacío Cuántico

`HarmonicVacuum` genera fluctuaciones deterministas para coordenadas vacías:

```cpp
torch::Tensor Engine::get_state_at(const Coord3D& coord) {
    if (matter_map_.contains_coord(coord)) {
        return matter_map_.get_tensor(coord);  // Materia real
    }
    return vacuum_.get_fluctuation(coord, step_count_);  // Vacío cuántico
}
```

Estas fluctuaciones son:
- **Deterministas**: La misma coordenada siempre genera la misma fluctuación (para el mismo step)
- **No cero**: El vacío tiene energía y puede interactuar con la materia
- **Modificables**: Las celdas activas pueden influir en el vacío circundante

## Flujo de Evolución

```
1. Agregar partícula en (x, y)
   ↓
2. activate_neighborhood() → Activa 8 vecinos
   ↓
3. step_native() procesa todas las celdas activas
   ↓
4. Para cada celda activa:
   - get_state_at(coord) → Materia o vacío
   - build_batch_input() → Construye patch con vecinos
   - Modelo procesa patch → delta_psi
   ↓
5. Si delta_psi > umbral → Crea/actualiza partícula
   ↓
6. Si nueva partícula → activate_neighborhood() → Propaga actividad
```

## Implicaciones

### ✅ Ventajas

1. **Propagación automática**: Los efectos se propagan naturalmente a vecinos
2. **Vacío interactivo**: El vacío puede ser modificado por partículas cercanas
3. **Eficiencia**: Solo se procesa lo necesario (regiones activas)

### ⚠️ Consideraciones

1. **Radio de vecindario**: Actualmente es 1 (8 vecinos en 2D). Podría ser configurable.
2. **Propagación rápida**: Si hay muchas partículas, `active_region_` puede crecer rápidamente
3. **Vacío determinista**: Las fluctuaciones del vacío son deterministas (no aleatorias), lo que puede ser deseable o no según el caso de uso

## Futuro: Partículas Virtuales

Como menciona el usuario, en QFT el vacío genera partículas virtuales. En el futuro, podríamos implementar:

1. **Generación automática de partículas virtuales** desde el vacío
2. **Colapso de partículas virtuales** si tienen suficiente energía
3. **Interacción materia-vacío** más sofisticada

Por ahora, seguimos con la implementación actual que ya incluye:
- Vacío cuántico (fluctuaciones deterministas)
- Activación de vecindarios
- Procesamiento de celdas activas y sus vecinos

## Referencias

- [[NATIVE_ENGINE_WRAPPER]]: Wrapper Python para el motor C++
- [[TECHNICAL_ARCHITECTURE_V4]]: Arquitectura general del sistema
- `src/cpp_core/src/sparse_engine.cpp`: Implementación C++
- `src/cpp_core/include/sparse_engine.h`: Interfaz C++

