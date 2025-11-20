# Benchmark: Motor Python vs Motor C++ Nativo

## 📊 Resumen

Este documento describe el proceso y los resultados del benchmark comparativo entre el motor Python (`Aetheria_Motor`) y el motor C++ nativo (`NativeEngineWrapper`).

## 🎯 Objetivo

Comparar el rendimiento de ambos motores ejecutando el mismo experimento y midiendo:
- **Throughput**: Pasos por segundo (SPS)
- **Latencia**: Tiempo de ejecución total
- **Memoria**: Uso de RAM
- **Precisión**: Verificar que ambos motores producen resultados similares

## 🔧 Uso del Script

El script de benchmark está disponible en `scripts/benchmark_python_vs_native.py`:

```bash
# Usar con un experimento específico
python3 scripts/benchmark_python_vs_native.py \
    --experiment EXPERIMENT_NAME \
    --steps 100 \
    --device cpu

# Ejemplo con más pasos y GPU
python3 scripts/benchmark_python_vs_native.py \
    --experiment UNET_32ch_D5_LR2e-5 \
    --steps 500 \
    --device cuda \
    --output benchmark_report_unet.md
```

### Opciones

- `--experiment`: Nombre del experimento (requerido)
- `--steps`: Número de pasos a ejecutar (default: 100)
- `--warmup`: Pasos de warm-up (default: 10)
- `--device`: Device (`cpu`/`cuda`) - default: auto-detección
- `--output`: Ruta del reporte (default: `benchmark_report_EXPERIMENT.md`)

### Requisitos

1. **Experimento con checkpoint**: El experimento debe tener al menos un checkpoint guardado en `output/checkpoints/EXPERIMENT_NAME/`
2. **Configuración del experimento**: Debe existir `output/experiments/EXPERIMENT_NAME/config.json`
3. **Motor nativo compilado**: El módulo `atheria_core` debe estar compilado (ver `docs/40_Experiments/PHASE_2_SETUP_LOG.md`)
4. **Modelo TorchScript**: El motor nativo requiere un modelo exportado a TorchScript (se exporta automáticamente si no existe usando la función mejorada)

### Notas Importantes

- **Exportación automática mejorada**: Si no existe un modelo TorchScript, el benchmark lo exportará automáticamente usando el tamaño completo del grid de inferencia (no patches pequeños), crucial para modelos UNet.
- **Manejo de memoria**: El script limpia memoria entre benchmarks para obtener mediciones precisas.
- **Warm-up**: Los pasos de warm-up permiten que el motor "se caliente" antes de medir rendimiento real.

## 📋 Métricas Medidas

### Motor Python

- **Tiempo de carga**: Tiempo para cargar el modelo desde checkpoint
- **Tiempo de inicialización**: Tiempo para crear motor y estado cuántico
- **Tiempo de pasos**: Tiempo para ejecutar N pasos de simulación
- **Pasos/segundo**: Throughput calculado
- **Memoria**: Uso de RAM antes/durante/después

### Motor C++ Nativo

- **Tiempo de carga**: Tiempo para exportar/cargar modelo TorchScript
- **Tiempo de inicialización**: Tiempo para inicializar wrapper y motor nativo
- **Tiempo de pasos**: Tiempo para ejecutar N pasos (todo en C++)
- **Pasos/segundo**: Throughput calculado
- **Memoria**: Uso de RAM antes/durante/después

### Comparación

- **Speedup**: Mejora de velocidad (nativo vs Python)
- **Overhead de memoria**: Diferencia en uso de RAM
- **Precisión**: Diferencia en energía final (para verificar consistencia)

## 📊 Resultados Esperados

### Escenarios de Benchmark

1. **CPU Mode**:
   - El motor nativo debería ser más rápido al ejecutar la lógica core en C++
   - Overhead de bindings puede afectar para operaciones pequeñas
   - Ventajas más claras en operaciones intensivas

2. **GPU Mode**:
   - Ambos motores usan CUDA para el modelo
   - El motor nativo puede optimizar mejor las operaciones dispersas
   - Diferencia de rendimiento depende de la complejidad del modelo

3. **Modelos Pequeños**:
   - Overhead de bindings puede dominar
   - Diferencia de rendimiento menor

4. **Modelos Grandes**:
   - Ventajas del motor nativo más claras
   - Mejor escalabilidad

## 🔍 Interpretación de Resultados

### Speedup < 1.0x
- El motor Python es más rápido
- Posible overhead de bindings C++/Python
- Normal para operaciones pequeñas o modelos simples

### Speedup ~1.0x
- Rendimiento similar
- El overhead de bindings compensa las optimizaciones
- Considerar otros factores (memoria, escalabilidad)

### Speedup > 1.0x
- El motor nativo es más rápido
- Ventajas del código C++ optimizado
- Escalabilidad mejor con modelos grandes

### Precisión (Diferencia de Energía)

- **< 1%**: ✅ Excelente precisión
- **1-5%**: ⚠️ Aceptable (puede ser por diferencias numéricas)
- **> 5%**: ❌ Problema de precisión (investigar diferencias de implementación)

## 📝 Reporte Generado

El script genera un reporte en Markdown con:

1. **Resumen ejecutivo**: Speedup, tiempo total, memoria
2. **Tabla comparativa**: Métricas lado a lado
3. **Análisis detallado**: Interpretación de resultados
4. **Detalles técnicos**: Tiempos de carga, inicialización, etc.

## 🚀 Próximos Pasos

1. **Ejecutar benchmark con diferentes experimentos**:
   - Modelos pequeños (MLP)
   - Modelos medianos (UNet 32ch)
   - Modelos grandes (UNet 64ch, ConvLSTM)

2. **Comparar en diferentes devices**:
   - CPU mode
   - GPU mode (si disponible)

3. **Optimizaciones adicionales**:
   - Optimizar conversión disperso ↔ denso
   - Reducir overhead de bindings
   - Optimizaciones específicas del modelo

## 🔗 Referencias

- `scripts/benchmark_python_vs_native.py`: Script de benchmark
- `src/engines/qca_engine.py`: Motor Python
- `src/engines/native_engine_wrapper.py`: Wrapper del motor nativo
- `src/cpp_core/`: Implementación C++ del motor nativo
- [[PHASE_2_MIGRATION_TO_NATIVE]]: Guía de migración al motor nativo

---

## 🚨 Estado Actual

**Estado:** ⏳ Benchmark creado, pendiente de ejecución con experimento válido

### Estado Actual del Benchmark

**Script creado:** ✅ `scripts/benchmark_python_vs_native.py`  
**Documentación:** ✅ Completa  
**Ejecución:** ⏳ Pendiente - requiere experimento con:
- `config.json` en `output/experiments/EXPERIMENT_NAME/`
- Checkpoint en `output/checkpoints/EXPERIMENT_NAME/*.pth`

### Ejecutar Benchmark

Para ejecutar el benchmark cuando haya un experimento válido:

```bash
# CPU mode (más rápido para pruebas)
python3 scripts/benchmark_python_vs_native.py \
    --experiment EXPERIMENT_NAME \
    --steps 100 \
    --device cpu

# GPU mode (si CUDA está disponible)
python3 scripts/benchmark_python_vs_native.py \
    --experiment EXPERIMENT_NAME \
    --steps 100 \
    --device cuda
```

El script generará un reporte en Markdown: `benchmark_report_EXPERIMENT_NAME.md`

### Por Qué el Motor Nativo No Se Usa Automáticamente

El motor nativo C++ está **disponible** (`NATIVE_AVAILABLE = True`), pero requiere:

1. **Modelo TorchScript exportado**: El motor nativo necesita un modelo exportado a `.pt` (TorchScript)
2. **Exportación automática**: El servidor intenta exportar automáticamente cuando carga un experimento, pero puede fallar si:
   - No hay checkpoint disponible
   - El modelo no es compatible con TorchScript
   - Hay errores en la exportación

### Cómo Forzar el Uso del Motor Nativo

1. **Exportar modelo manualmente**:
   ```bash
   python scripts/test_native_engine.py --experiment EXPERIMENT_NAME --export-only
   ```

2. **Verificar si hay modelo JIT**:
   ```bash
   ls output/torchscript_models/EXPERIMENT_NAME/model.pt
   ```

3. **Revisar logs del servidor** cuando cargas un experimento para ver si exporta el modelo automáticamente.

---

**Última actualización:** 2024-11-20  
**Próximo paso:** Ejecutar benchmark cuando haya modelos disponibles

