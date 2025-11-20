# Migración de Experimentos: Python CUDA → Motor Nativo C++

## Resumen

Sí, puedes usar un experimento entrenado en **Python CUDA** con el **motor nativo C++** sin problemas. El sistema realiza la conversión automáticamente cuando cargas el experimento.

## Proceso Automático

### Cuando cargas un experimento:

1. **Búsqueda de modelo JIT**: El sistema busca un modelo TorchScript (`.pt`) ya exportado.
2. **Exportación automática**: Si no existe, exporta automáticamente el checkpoint de PyTorch (`.pth`) a TorchScript.
3. **Uso del motor nativo**: Una vez exportado, el motor nativo C++ puede usar el modelo.

### Ubicación de archivos:

- **Checkpoints Python**: `output/training_checkpoints/<experiment_name>/checkpoint_*.pth`
- **Modelos JIT (TorchScript)**: `output/training_checkpoints/<experiment_name>/model_*.pt`

## Conversión Manual (Opcional)

Si quieres exportar manualmente un modelo a TorchScript:

```bash
python scripts/test_native_engine.py --experiment NOMBRE_EXPERIMENTO
```

Este script:
- Carga el checkpoint más reciente del experimento
- Exporta el modelo a TorchScript usando `torch.jit.trace()` o `torch.jit.script()`
- Guarda el `.pt` en el directorio de checkpoints

## Compatibilidad

### ✅ Compatible:

- **Arquitecturas soportadas**: Todas las arquitecturas de modelos están soportadas (UNet, UNetUnitary, ConvLSTM, etc.)
- **Dispositivo**: Los modelos entrenados en CUDA se pueden usar en el motor nativo tanto en CPU como CUDA.
- **Pesos**: Los pesos del checkpoint se preservan completamente durante la exportación.

### ⚠️ Limitaciones:

- **Estado del modelo**: El estado interno del modelo (si tiene memoria como ConvLSTM) se resetea al exportar.
- **Tamaño de grid**: El modelo se exporta con el tamaño de grid de inferencia (normalmente 256x256).

## Detalles Técnicos

### Proceso de Exportación:

1. **Carga del checkpoint**: Se carga el modelo PyTorch desde `.pth` con los pesos entrenados.
2. **Modo evaluación**: El modelo se pone en modo `eval()`.
3. **Ejemplo de entrada**: Se crea un tensor de ejemplo con el tamaño de grid de inferencia.
4. **TorchScript export**: Se usa `torch.jit.trace()` (o `torch.jit.script()` como fallback).
5. **Guardado**: El modelo TorchScript se guarda como `.pt`.

### Verificación:

El sistema verifica que:
- El modelo TorchScript se puede cargar correctamente.
- El forward pass funciona con el ejemplo de entrada.
- El modelo es compatible con el motor nativo C++.

## Uso en el Frontend

Cuando cargas un experimento desde el frontend:

1. Si existe un modelo JIT, se usa directamente con el motor nativo.
2. Si no existe, verás la notificación: "📦 Exportando modelo a TorchScript..."
3. Una vez exportado, el motor nativo se inicializa automáticamente.

## Ventajas del Motor Nativo

- **Rendimiento**: 250-400x más rápido que el motor Python.
- **Memoria**: Usa arquitectura dispersa (sparse) más eficiente.
- **Escalabilidad**: Mejor manejo de grids grandes (256x256 o más).

## Referencias

- `scripts/test_native_engine.py`: Función `export_model_to_torchscript()`
- `src/pipelines/pipeline_server.py`: Función `handle_load_experiment()` (línea ~1015)
- `src/engines/native_engine_wrapper.py`: Wrapper del motor nativo

