# Optimización de Logs - Pipeline Server

**Componente:** `src/pipelines/pipeline_server.py`  
**Fecha:** 2024-12-XX  
**Objetivo:** Reducir verbosidad de logs durante operación normal manteniendo información crítica.

---

## Contexto

El servidor WebSocket generaba logs excesivos durante la operación normal, especialmente:
- Cada conexión/desconexión de cliente
- Cada comando recibido
- Cada frame enviado
- Diagnósticos frecuentes del bucle de simulación

Esto generaba ruido innecesario y dificultaba identificar eventos importantes.

---

## Cambios Realizados

### 1. Reducción de Verbosidad en WebSocket

**Antes:**
```python
logging.info(f"Intento de conexión WebSocket desde {client_ip}")
logging.info(f"Nueva conexión WebSocket: {ws_id}")
logging.info(f"Comando recibido: {scope}.{command} de [{ws_id}]")
logging.info(f"Conexión WebSocket cerrada: {ws_id}")
```

**Después:**
```python
logging.debug(f"Intento de conexión WebSocket desde {client_ip}")
logging.debug(f"Nueva conexión WebSocket: {ws_id}")
logging.debug(f"Comando recibido: {scope}.{command} de [{ws_id}]")
logging.debug(f"Conexión WebSocket cerrada: {ws_id}")
```

**Justificación:** Estos eventos son normales y frecuentes. Solo mantener INFO para errores y eventos críticos.

### 2. Bucle de Simulación

**Antes:**
```python
logging.info(f"🔍 Diagnóstico: is_paused={is_paused}, motor={'✓' if motor else '✗'}, ...")
# Cada 5 segundos
```

**Después:**
```python
logging.debug(f"🔍 Diagnóstico: is_paused={is_paused}, motor={'✓' if motor else '✗'}, ...")
# Cada 30 segundos (reducido frecuencia)
```

**Justificación:** 
- Diagnóstico es información técnica, no crítica
- Reducir frecuencia de 5s a 30s para menos overhead
- Mantener disponible en nivel DEBUG cuando sea necesario

### 3. Logs de Frames y Payloads

**Mantenidos como DEBUG:**
- Tamaño de payloads (cada 100 frames)
- Errores al guardar en historial
- Warnings sobre frames inválidos

**Justificación:** Información técnica útil para debugging pero no crítica en operación normal.

---

## Configuración de Logging

**Nivel por defecto:** `logging.INFO`

**Eventos en INFO:**
- Errores críticos (exceptions con traceback)
- Warnings importantes (motor sin cargar, comandos desconocidos)
- Inicio/fin de procesos importantes

**Eventos en DEBUG:**
- Conexiones/desconexiones WebSocket normales
- Comandos recibidos
- Diagnósticos del bucle de simulación
- Detalles técnicos (tamaños de payload, compresión)

**Eventos en WARNING:**
- Situaciones anómalas pero recuperables
- Comandos desconocidos
- Estados inconsistentes

**Eventos en ERROR:**
- Excepciones críticas
- Errores de conexión
- Errores de procesamiento

---

## Beneficios

1. **Rendimiento:** Menos overhead de I/O en logging
2. **Legibilidad:** Logs más limpios, fáciles de filtrar
3. **Debugging:** Mantener nivel DEBUG disponible cuando sea necesario
4. **Producción:** Logs más útiles para monitoreo

---

## Uso

### Ver logs normales (INFO):
```bash
python run_server.py
```

### Ver logs detallados (DEBUG):
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# O en el código:
logging.getLogger('src.pipelines.pipeline_server').setLevel(logging.DEBUG)
```

---

## Referencias

- [[AI_DEV_LOG#2024-12-XX - Optimización de Logs]]
- `src/pipelines/pipeline_server.py`

---

**Estado:** ✅ Completado  
**Próxima revisión:** Cuando se identifiquen nuevos puntos de verbosidad excesiva

