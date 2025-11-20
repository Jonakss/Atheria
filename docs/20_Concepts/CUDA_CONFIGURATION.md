# Configuración de CUDA para Atheria 4

## 🔍 Diagnóstico de CUDA

### Problema Común: Error 101 (Invalid Device Ordinal)

**Síntoma:**
```
CUDA initialization: Unexpected error from cudaGetDeviceCount().
Error 101: invalid device ordinal
```

**Causas Posibles:**
1. **PyTorch compilado con CUDA pero sin GPU disponible**: PyTorch fue compilado con soporte CUDA, pero no hay dispositivos CUDA detectables en el sistema.
2. **Drivers de CUDA no instalados o desactualizados**: Los drivers de NVIDIA no están instalados o son incompatibles.
3. **Problema de compatibilidad entre PyTorch y CUDA runtime**: Versión de PyTorch incompatible con la versión de CUDA instalada.
4. **CUDA_VISIBLE_DEVICES configurado incorrectamente**: La variable de entorno limita los dispositivos disponibles.

### Verificar Estado de CUDA

```bash
# 1. Verificar que hay GPUs disponibles
nvidia-smi

# 2. Verificar versión de PyTorch y CUDA
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA built:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

# 3. Verificar drivers de CUDA
nvcc --version
```

## 🛠️ Soluciones

### 1. Forzar CPU Mode (Si No Hay GPU)

Si no hay GPU disponible, el sistema automáticamente usa CPU. Para forzarlo explícitamente:

```bash
# Variable de entorno para PyTorch
export ATHERIA_FORCE_DEVICE=cpu

# Variable de entorno para motor nativo
export ATHERIA_NATIVE_DEVICE=cpu
```

### 2. Forzar CUDA Mode (Si Hay GPU Pero No Se Detecta)

Si hay GPU pero PyTorch no la detecta:

```bash
# Forzar CUDA para PyTorch
export ATHERIA_FORCE_DEVICE=cuda

# Forzar CUDA para motor nativo
export ATHERIA_NATIVE_DEVICE=cuda

# Forzar dispositivo específico
export CUDA_VISIBLE_DEVICES=0
```

### 3. Instalar/Actualizar Drivers de CUDA

Si `nvidia-smi` no funciona o no detecta GPUs:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-xxx  # Reemplazar xxx con versión disponible

# Verificar instalación
nvidia-smi
```

### 4. Reinstalar PyTorch con CUDA Correcto

Si PyTorch está compilado con CUDA pero no funciona:

```bash
# Desinstalar PyTorch actual
pip uninstall torch torchvision

# Instalar PyTorch con CUDA específico
# Para CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Para CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para CPU solamente:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 📊 Configuración en Atheria 4

### Detección Automática

El sistema intenta detectar CUDA automáticamente en este orden:

1. **Verificación de `torch.cuda.is_available()`**: PyTorch reporta si CUDA está disponible.
2. **Verificación de `torch.cuda.device_count()`**: Verifica que haya dispositivos disponibles.
3. **Prueba de Tensor CUDA**: Crea un tensor pequeño en CUDA para verificar que funciona.

Si alguna de estas verificaciones falla, el sistema usa CPU como fallback.

### Variables de Entorno

- **`ATHERIA_FORCE_DEVICE`**: Fuerza el device para PyTorch (`cpu` o `cuda`)
- **`ATHERIA_NATIVE_DEVICE`**: Fuerza el device para el motor nativo C++ (`cpu`, `cuda`, o `auto`)
- **`CUDA_VISIBLE_DEVICES`**: Limita qué GPUs son visibles para CUDA (ej: `0` para solo GPU 0)

### Motor Nativo C++

El motor nativo C++ también soporta CUDA a través de LibTorch. Para usar CUDA con el motor nativo:

1. **Asegurar que LibTorch esté compilado con CUDA**: El módulo `atheria_core` debe estar compilado con soporte CUDA.
2. **Configurar device**: Usar `device='cuda'` o `device=None` (auto-detección) al inicializar `NativeEngineWrapper`.
3. **Verificar importación**: Si hay problemas de CUDA runtime al importar `atheria_core`, el sistema automáticamente intenta CPU mode.

## 🔧 Troubleshooting

### Error: "CUDA available: False" pero PyTorch tiene CUDA

**Diagnóstico:**
```bash
python3 -c "import torch; print(torch.version.cuda)"  # Debe mostrar versión
nvidia-smi  # Debe mostrar GPUs disponibles
```

**Solución:**
1. Verificar que los drivers de NVIDIA estén instalados.
2. Verificar que CUDA runtime esté instalado y sea compatible.
3. Intentar forzar CUDA: `export ATHERIA_FORCE_DEVICE=cuda`

### Error: "Error 101: invalid device ordinal" con GPU disponible

**Causa:** Incompatibilidad entre PyTorch y CUDA runtime.

**Solución:**
1. Verificar versión de CUDA: `nvcc --version`
2. Reinstalar PyTorch con versión compatible de CUDA.
3. Verificar `CUDA_VISIBLE_DEVICES`: `echo $CUDA_VISIBLE_DEVICES`

### Motor Nativo No Usa CUDA

**Verificar:**
```bash
# Verificar device del motor nativo en logs
grep "Motor nativo" output/logs/*.log

# Verificar variable de entorno
echo $ATHERIA_NATIVE_DEVICE
```

**Solución:**
```bash
export ATHERIA_NATIVE_DEVICE=cuda
python3 run_server.py
```

## 📝 Notas Importantes

- **CPU Mode es Funcional**: El sistema funciona perfectamente en CPU, solo es más lento.
- **Detección Automática**: Por defecto, el sistema detecta automáticamente el mejor dispositivo disponible.
- **Fallback Seguro**: Si CUDA falla, el sistema automáticamente usa CPU sin interrumpir la ejecución.
- **Motor Nativo**: El motor nativo C++ también soporta CUDA, proporcionando aceleración adicional.

## 🔗 Referencias

- [[NATIVE_ENGINE_DEVICE_CONFIG]]: Configuración específica del motor nativo
- [[PHASE_2_CUDA_RUNTIME_FIX]]: Fix específico para problemas de CUDA runtime
- `src/config.py`: Implementación de detección de CUDA
- `src/engines/native_engine_wrapper.py`: Wrapper del motor nativo con soporte CUDA

---

**Última actualización:** 2024-11-20

