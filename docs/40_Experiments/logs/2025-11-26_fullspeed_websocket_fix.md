# 2025-11-26: Fix Saturación WebSocket en Modo Full Speed

**Fecha:** 2025-11-26
**Autor:** Antigravity (Google Deepmind)
**Tipo:** `fix`
**Componentes:** `src/pipelines/core/simulation_loop.py`

---

## 📝 Resumen Ejecutivo

Se corrigió un bug crítico donde el modo "full speed" (`steps_interval = -1`) seguía enviando frames de visualización, actualizaciones de estado y logs al frontend vía WebSocket, saturando la conexión y dando la impresión de que la simulación corría "en tiempo real" cuando debería ejecutarse a máxima velocidad sin overhead de comunicación.

## 🐛 Problema Identificado

El usuario reportó que aunque desactivara el "live feed" o configurara `steps_interval = -1`, la visualización seguía actualizándose y mostraba métricas como "0.03 (Ahorro!)" en rojo, indicando transferencia de datos continua.

### Causa Raíz

En `src/pipelines/core/simulation_loop.py` había **tres puntos críticos** donde se enviaban datos sin verificar si `steps_interval == -1`:

1. **Líneas 210-216**: La condición `should_send_frame` tenía un bug lógico donde `steps_interval == -1` caía en el bloque `else`, calculando `steps_interval_counter >= -1`, que **SIEMPRE** es `True` → frames enviados continuamente.

2. **Líneas 469-488**: El `state_update` throttled se enviaba cada `STATE_UPDATE_INTERVAL` segundos **siempre**, sin verificar el modo full speed.

3. **Líneas 492-503**: Los logs de simulación se enviaban cada 100 pasos **siempre**, sin verificar el modo full speed.

**Resultado:** En modo full speed, el backend enviaba ~2-10 mensajes/segundo saturando el WebSocket innecesariamente.

## 🔧 Solución Implementada

### 1. Fix: `should_send_frame` para Full Speed (Líneas 207-220)

Agregada verificación explícita para `steps_interval == -1` ANTES del bloque `else`:

```python
# ANTES
if steps_interval == 0:
    should_send_frame = (g_state['last_frame_sent_step'] == -1)
else:  # ❌ PROBLEMA: -1 cae aquí
    should_send_frame = (steps_interval_counter >= steps_interval) or ...

# DESPUÉS
if steps_interval == -1:
    # Modo fullspeed: NUNCA enviar frames
    should_send_frame = False
elif steps_interval == 0:
    # Modo manual: Solo el frame inicial
    should_send_frame = (g_state['last_frame_sent_step'] == -1)
else:
    # Modo automático: cada N pasos
    should_send_frame = (steps_interval_counter >= steps_interval) or ...
```

### 2. Fix: State Update Throttling (Línea 471)

```python
# ANTES
if time_since_last_update >= STATE_UPDATE_INTERVAL:

# DESPUÉS
if steps_interval != -1 and time_since_last_update >= STATE_UPDATE_INTERVAL:
```

### 3. Fix: Simulation Log Throttling (Línea 494)

```python
# ANTES
if updated_step % 100 == 0:

# DESPUÉS
if steps_interval != -1 and updated_step % 100 == 0:
```

## ✅ Resultado

| Modo | Antes | Ahora |
|------|-------|-------|
| **Full Speed (-1)** | Enviaba frames + updates + logs (saturación) | NO envía nada (máximo rendimiento) ✅ |
| **Manual (0)** | Funcionaba correctamente | Sin cambios |
| **Automático (N > 0)** | Funcionaba correctamente | Sin cambios |

- ✅ Modo full speed ejecuta pasos a máxima velocidad SIN overhead de WebSocket
- ✅ No se satura la conexión con datos innecesarios
- ✅ El frontend muestra correctamente que no hay visualización activa
- ✅ Ganancia de rendimiento: eliminado 100% del overhead de comunicación en modo full speed

## 🔗 Archivos Afectados

- [`src/pipelines/core/simulation_loop.py`](file:///home/jonathan.correa/Projects/Atheria/src/pipelines/core/simulation_loop.py#L207-L504) - Tres fixes críticos

## 📦 Commits

- `2ec69cc` - fix: prevenir envío de frames/updates en modo full speed (steps_interval=-1) [version:bump:patch]
