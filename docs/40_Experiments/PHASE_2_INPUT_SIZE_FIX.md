# Phase 2: Corrección de Tamaño de Input en Motor Nativo

**Fecha:** 2025-01-20  
**Estado:** En progreso

## Problema Identificado

El motor nativo C++ estaba construyendo inputs de tamaño incorrecto para los modelos UNet. El error específico era:

```
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 4 but got size 5 for tensor number 1 in the list.
```

### Causa Raíz

1. **Modelo entrenado con grid completo:** Los modelos UNet fueron entrenados con inputs de tamaño completo del grid (típicamente 64x64 o 128x128), no con patches pequeños.

2. **Skip connections:** La arquitectura U-Net tiene skip connections (`torch.cat([u2, x1], dim=1)`) que requieren que las dimensiones espaciales coincidan exactamente en diferentes niveles. Si el modelo fue entrenado con 64x64, espera ese tamaño exacto.

3. **Parches pequeños:** El código C++ estaba intentando usar patches de 3x3 o 5x5 alrededor de cada partícula, lo cual es ineficiente pero conceptualmente correcto para un motor disperso. Sin embargo, esto causa incompatibilidad con los skip connections.

## Cambios Implementados

### 1. Aumento de tamaño de patch

- **Antes:** Patch de 3x3 (insuficiente para MaxPool2d dos veces)
- **Intermedio:** Patch de 5x5 (funciona con MaxPool2d pero falla con skip connections)
- **Actual:** Preparado para usar tamaño completo del grid (64x64 por defecto)

### 2. Manejo de modelos ConvLSTM

Se añadió manejo de argumentos opcionales `h_t` y `c_t` para modelos que requieren memoria temporal:

```cpp
try {
    batch_output = model_.forward(inputs).toTensor();
} catch (const std::exception& e) {
    // Si falla, el modelo puede requerir h_t y c_t explícitos
    inputs.push_back(torch::IValue());  // h_t = None
    inputs.push_back(torch::IValue());  // c_t = None
    auto output_tuple = model_.forward(inputs).toTuple();
    batch_output = output_tuple->elements()[0].toTensor();
}
```

### 3. Mejora del manejo de CUDA Runtime

Se mejoró la detección y manejo de problemas de CUDA runtime en `native_engine_wrapper.py`:

- Detección automática de problemas de librerías CUDA
- Forzar CPU mode cuando hay problemas
- Configurar `CUDA_VISIBLE_DEVICES=''` automáticamente

## Problema Pendiente

**El modelo requiere el tamaño completo del grid:** Para que los skip connections funcionen correctamente, necesitamos usar el tamaño completo del grid (64x64 o el tamaño con el que fue entrenado), no patches pequeños.

### Opciones de Solución

1. **Usar grid completo (ineficiente pero funciona):**
   - Construir un input del tamaño completo del grid para cada partícula
   - Centrar el input en la posición de la partícula
   - Funciona pero es muy ineficiente en memoria

2. **Re-entrenar modelo con patches (óptimo a largo plazo):**
   - Modificar la arquitectura para que funcione con patches de tamaño fijo
   - Requiere re-entrenamiento pero es la solución más eficiente

3. **Padding dinámico (complejo):**
   - Usar padding para ajustar tamaños en los skip connections
   - Requiere modificar el modelo o hacer post-procesamiento

## Próximos Pasos

1. ✅ Corregir tamaño de input mínimo (5x5 → tamaño completo del grid)
2. ✅ Implementar uso de tamaño completo del grid desde configuración (agregar grid_size al constructor)
3. ✅ Usar grid_size_ en build_batch_input() para construir inputs del tamaño correcto
4. ⏳ Optimizar para reducir uso de memoria cuando se usa grid completo
5. 🔄 Considerar re-entrenamiento con arquitectura compatible con patches (futuro)
6. ⏳ Probar con modelos reales en CPU y GPU

## Referencias

- `src/cpp_core/src/sparse_engine.cpp::build_batch_input()`
- `src/models/unet_convlstm.py::forward()` - Skip connections en líneas 166, 170
- `docs/40_Experiments/PHASE_2_MIGRATION_TO_NATIVE.md`

