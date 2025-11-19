# Optimizaciones de GPU y CPU

Este documento describe las optimizaciones implementadas para mejorar el uso de GPU y CPU en Aetheria.

## Optimizaciones Implementadas

### 1. **Gestión de Memoria GPU**

#### Limpieza Automática de Cache
- La cache de GPU se limpia automáticamente cada 100 pasos de simulación
- Evita que la memoria se fragmente durante simulaciones largas
- Se sincroniza con CUDA para asegurar limpieza completa

#### Optimizaciones CUDA
```python
torch.backends.cudnn.benchmark = True  # Optimiza para tamaños fijos
torch.backends.cudnn.deterministic = False  # Permite optimizaciones no deterministas
```

### 2. **Modo de Inferencia Optimizado**

#### `torch.inference_mode()` vs `torch.no_grad()`
- Se usa `torch.inference_mode()` en lugar de `torch.no_grad()` para mejor rendimiento
- `inference_mode()` es más rápido y tiene menos overhead que `no_grad()`
- Disponible en PyTorch 1.9+

#### Modelo en Modo Evaluación
- El modelo se mantiene siempre en modo `.eval()` durante inferencia
- Desactiva dropout, batch normalization en modo train, etc.

### 3. **Compilación de Modelos (PyTorch 2.0+)**

#### `torch.compile()`
- Si está disponible, el modelo se compila con `torch.compile()` para inferencia optimizada
- Modo `reduce-overhead`: Optimiza para reducir overhead de llamadas
- Puede mejorar el rendimiento 2-3x en algunos casos

### 4. **Optimización de Transferencias CPU↔GPU**

#### Evitar Transferencias Innecesarias
- Los tensores se mantienen en GPU durante la simulación
- Solo se mueven a CPU cuando es necesario para visualización
- Función `should_keep_on_gpu()` decide automáticamente si mantener en GPU

#### Batch Transfers
- Se usa `move_to_cpu_batch()` para mover múltiples tensores eficientemente
- Reduce el número de transferencias individuales

### 5. **Optimización de Visualizaciones**

#### Lazy CPU Transfer
- Los datos solo se mueven a CPU cuando se van a usar para visualización
- Si `live_feed_enabled` está desactivado, no se mueven datos innecesariamente

#### Downsampling para Visualización
- Se puede aplicar downsampling antes de transferir a CPU
- Reduce el tamaño de datos transferidos

## Uso

### Inicialización Automática

El optimizador se inicializa automáticamente cuando se crea un `Aetheria_Motor`:

```python
motor = Aetheria_Motor(model, grid_size, d_state, device, cfg=config)
# El optimizador se configura automáticamente
```

### Estadísticas de Memoria

Puedes obtener estadísticas de memoria en cualquier momento:

```python
stats = motor.optimizer.get_memory_stats()
# stats = {
#     'device': 'cuda:0',
#     'step_count': 1234,
#     'gpu_allocated_mb': 256.5,
#     'gpu_reserved_mb': 512.0,
#     'gpu_max_allocated_mb': 768.0
# }
```

## Configuración

### Intervalo de Limpieza de Cache

Por defecto, la cache se limpia cada 100 pasos. Puedes cambiarlo:

```python
motor.optimizer.empty_cache_interval = 50  # Limpiar cada 50 pasos
```

### Threshold de Tamaño para CPU

Por defecto, tensores < 10MB se mantienen en GPU. Puedes cambiarlo:

```python
should_keep = motor.optimizer.should_keep_on_gpu(tensor, size_threshold_mb=5.0)
```

## Mejoras de Rendimiento Esperadas

- **GPU**: 10-30% mejora en throughput de inferencia
- **Memoria**: Reducción de fragmentación, mejor uso de memoria
- **CPU**: Menos overhead de gestión de gradientes
- **Transferencias**: Reducción de 50-70% en transferencias innecesarias

## Notas

- Las optimizaciones son automáticas y no requieren cambios en el código de uso
- Compatible con CPU (aunque las optimizaciones GPU no aplican)
- Compatible con modelos con y sin memoria (ConvLSTM)
- Funciona con todos los tipos de modelos (UNet, MLP, DEEP_QCA, etc.)

