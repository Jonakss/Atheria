---
title: Cómo Ejecutar Atheria 4
type: guide
status: active
tags: [guide, quick-start, cli, development]
created: 2024-11-19
updated: 2025-11-20
aliases: [Quick Start, Getting Started, How to Run]
related: [[30_Components/CLI_TOOL|CLI Tool]], [[30_Components/Native_Engine_Core|Motor Nativo]], [[OBSIDIAN_SETUP|Setup Obsidian]]
---

# Cómo Ejecutar Atheria 4

> **Guía rápida** para ejecutar el servidor backend y el frontend usando el CLI o comandos directos.

---

## 🚀 Opción Recomendada: CLI Tool

El CLI `atheria` (alias `ath`) simplifica el flujo de desarrollo completo.

### Instalación del CLI

Después de instalar el paquete en modo desarrollo:

```bash
cd /home/jonathan.correa/Projects/Atheria
source ath_venv/bin/activate
pip install -e .
```

El comando `atheria` estará disponible globalmente.

### Uso del CLI

#### Desarrollo Completo (Build + Install + Run)

```bash
# Sin frontend (solo WebSocket API)
atheria dev

# Con frontend estático
atheria dev --frontend

# Puerto personalizado
atheria dev --port 8080 --frontend
```

#### Comandos Individuales

```bash
# Solo compilar extensiones C++
atheria build

# Solo instalar paquete
atheria install

# Solo ejecutar servidor
atheria run                  # Sin frontend
atheria run --frontend       # Con frontend

# Limpiar archivos de build
atheria clean
```

**👉 Ver documentación completa:** [[30_Components/CLI_TOOL|CLI Tool]]

---

## 📦 Instalación Manual

### 1. Backend (Servidor Python)

#### Activar el entorno virtual

```bash
cd /home/jonathan.correa/Projects/Atheria
source ath_venv/bin/activate  # En Windows: ath_venv\Scripts\activate
```

#### Instalar dependencias

```bash
pip install -r requirements.txt
```

#### Compilar extensiones C++ (Motor Nativo)

```bash
python3 setup.py build_ext --inplace
pip install -e .  # Instalar en modo desarrollo (habilita CLI)
```

#### Ejecutar el servidor

```bash
# Con frontend estático
python3 run_server.py

# Solo WebSocket API (sin frontend)
ATHERIA_NO_FRONTEND=1 python3 run_server.py

# Puerto personalizado
python3 run_server.py --port 8080
```

El servidor se iniciará en:
- **URL**: `http://localhost:8000` (o el puerto especificado)
- **Puerto por defecto**: `8000` (configurable en `src/config.py`)

### 2. Frontend (React)

**Nota:** El frontend se sirve automáticamente desde el backend cuando se ejecuta `run_server.py` (a menos que uses `--no-frontend` o `ATHERIA_NO_FRONTEND=1`).

#### Para desarrollo frontend separado

```bash
cd frontend
npm install
npm run dev
```

El frontend se iniciará en:
- **URL**: `http://localhost:5173` (Vite por defecto)
- **Puerto**: `5173` (configurable en `frontend/vite.config.ts`)

---

## ⚙️ Configuración

### Motor de Simulación

Puedes cambiar entre motor Python y motor nativo C++ desde la UI (header del dashboard) o configurarlo:

```python
# En src/config.py
USE_NATIVE_ENGINE = True  # Por defecto: True (máximo rendimiento)
```

**Rendimiento:**
- **Motor Nativo C++**: ~10,000 steps/segundo (recomendado)
- **Motor Python**: ~100-500 steps/segundo (más flexible)

**👉 Ver documentación:** [[30_Components/Native_Engine_Core|Motor Nativo C++]]

### Puertos

**Backend** (`src/config.py`):
```python
LAB_SERVER_HOST = '0.0.0.0'
LAB_SERVER_PORT = 8000
```

**Frontend** (`frontend/vite.config.ts` - solo si ejecutas frontend separado):
```typescript
server: {
  port: 5173,
  // ...
}
```

---

## 🔧 Workflow Recomendado

### Desarrollo Backend

```bash
# 1. Activar entorno
source ath_venv/bin/activate

# 2. Usar CLI para desarrollo completo
atheria dev --frontend

# 3. Solo recompilar después de cambios C++
atheria build

# 4. Limpiar cuando sea necesario
atheria clean
```

### Desarrollo Frontend

Si necesitas desarrollo frontend separado:

```bash
# Terminal 1: Backend (sin frontend)
atheria dev

# Terminal 2: Frontend (modo desarrollo con hot-reload)
cd frontend
npm run dev
```

Luego abre tu navegador en: `http://localhost:5173`

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'atheria_core'"

El motor nativo no está compilado. Compílalo:

```bash
atheria build
# O manualmente:
python3 setup.py build_ext --inplace
pip install -e .
```

**👉 Ver documentación:** [[30_Components/Native_Engine_Core|Motor Nativo C++]]

### Error: "Port 8000 already in use"

Opción 1: Cambiar puerto con CLI
```bash
atheria dev --port 8080
```

Opción 2: Cambiar en `src/config.py`:
```python
LAB_SERVER_PORT = 8001  # O cualquier otro puerto disponible
```

### Error: "Port 5173 already in use"

Solo aplica si ejecutas frontend separado. Cambia en `frontend/vite.config.ts`:

```typescript
server: {
  port: 5174,  // O cualquier otro puerto disponible
  // ...
}
```

### El frontend no se conecta al backend

1. Verifica que el backend esté corriendo en `http://localhost:8000`
2. Verifica la URL del WebSocket en la consola del navegador
3. Asegúrate de no tener CORS bloqueado
4. Revisa los logs del backend para errores de conexión

### Motor nativo no carga

1. Verifica que las extensiones C++ estén compiladas:
   ```bash
   ls src/atheria_core*.so  # Debe existir
   ```

2. Verifica la versión de LibTorch:
   - El motor nativo requiere LibTorch compatible
   - Verifica `docs/30_Components/Native_Engine_Core|Motor Nativo]]` para requisitos

3. Verifica los logs del backend:
   - Busca mensajes de error relacionados con `atheria_core`
   - Revisa `[[40_Experiments/NATIVE_ENGINE_PERFORMANCE_ISSUES|Problemas de Rendimiento]]`

---

## 🎯 Uso con Motor Nativo C++

### Verificación

Cuando cargas un experimento, el backend **automáticamente**:

1. Busca el modelo JIT (`.pt`) en el directorio del experimento
2. Si no existe, lo **exporta automáticamente** desde el checkpoint (`.pth`)
3. Carga el motor nativo con el modelo JIT
4. **¡Disfruta de ~10,000 steps/segundo!**

### Logs Esperados

En los logs del servidor, deberías ver:

```
✅ Motor nativo (C++) cargado exitosamente con modelo JIT
⚡ Motor nativo activo: device=cuda, grid_size=256
🚀 Rendimiento: ~10,000 steps/segundo
```

### Cambio de Motor

Puedes cambiar entre motores sin reiniciar:

1. Desde la UI: Click en el badge "Engine::native" → Seleccionar motor
2. Desde código: Cambia `USE_NATIVE_ENGINE` en `src/config.py` y reinicia

**👉 Ver documentación completa:** [[30_Components/Native_Engine_Core|Motor Nativo C++]]

---

## 📊 Optimizaciones Disponibles

### Control de Live Feed

- **Activo**: Calcula y envía visualizaciones en tiempo real
- **Desactivado**: Solo evoluciona la física sin visualizaciones
- **Modo Manual**: `steps_interval = 0` para control manual de actualizaciones

### Protocolo WebSocket Optimizado

- **MessagePack Binario**: Frames de visualización en formato binario (3-5x más compacto que JSON)
- **JSON para Comandos**: Solo comandos y metadatos usan JSON (pequeños y rápidos)

**👉 Ver documentación:** [[30_Components/WEB_SOCKET_PROTOCOL|Protocolo WebSocket]]

### Lazy Conversion y ROI

- **Lazy Conversion**: Conversión sparse→dense solo cuando se necesita
- **ROI Support**: Renderizado basado en Region of Interest para eficiencia

**👉 Ver documentación:** [[30_Components/Native_Engine_Core|Motor Nativo C++]]

---

## 🔗 Referencias Relacionadas

- [[30_Components/CLI_TOOL|CLI Tool]] - Documentación completa del CLI
- [[30_Components/Native_Engine_Core|Motor Nativo C++]] - Arquitectura y optimizaciones
- [[30_Components/WEB_SOCKET_PROTOCOL|Protocolo WebSocket]] - Protocolo binario vs JSON
- [[40_Experiments/AI_DEV_LOG|Log de Desarrollo AI]] - Cambios recientes y mejoras
- [[40_Experiments/NATIVE_ENGINE_PERFORMANCE_ISSUES|Problemas de Rendimiento]] - Troubleshooting motor nativo
- [[OBSIDIAN_SETUP|Setup Obsidian]] - Configuración del vault

---

## 🎓 Próximos Pasos

1. **Lee** [[10_core/PROGRESSIVE_LEARNING|Aprendizaje Progresivo]] para comenzar
2. **Prueba** [[40_Experiments/VISUALIZATION_TESTING|Pruebas de Visualización]]
3. **Experimenta** siguiendo [[40_Experiments/EXPERIMENTATION_GUIDE|Guía de Experimentación]]
4. **Explora** el [[30_Components/Native_Engine_Core|Motor Nativo]] para máximo rendimiento

---

*Última actualización: 2025-11-20*
