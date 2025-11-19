# Cómo Ejecutar Atheria 4

**Fecha**: 2024-11-19  
**Objetivo**: Guía rápida para ejecutar el servidor backend y el frontend.

## Comandos de Ejecución

### 1. Backend (Servidor Python)

#### Activar el entorno virtual

```bash
cd /home/jonathan.correa/Projects/Atheria
source ath_venv/bin/activate
```

#### Ejecutar el servidor

```bash
python run_server.py
```

El servidor se iniciará en:
- **URL**: `http://localhost:8000` (o `http://0.0.0.0:8000`)
- **Puerto**: `8000` (configurable en `src/config.py`)

### 2. Frontend (React)

#### Instalar dependencias (primera vez)

```bash
cd frontend
npm install
```

#### Ejecutar el frontend en modo desarrollo

```bash
cd frontend
npm run dev
```

El frontend se iniciará en:
- **URL**: `http://localhost:5173` (Vite por defecto)
- **Puerto**: `5173` (configurable en `frontend/vite.config.ts`)

### 3. Ejecutar Ambos (Recomendado)

Abre **dos terminales**:

#### Terminal 1: Backend
```bash
cd /home/jonathan.correa/Projects/Atheria
source ath_venv/bin/activate
python run_server.py
```

#### Terminal 2: Frontend
```bash
cd /home/jonathan.correa/Projects/Atheria/frontend
npm run dev
```

Luego abre tu navegador en: `http://localhost:5173`

## Configuración

### Motor Nativo (C++)

Por defecto, el motor nativo está **habilitado** para obtener máximo rendimiento (250-400x más rápido).

Si quieres deshabilitarlo (usar motor Python):

```python
# En src/config.py
USE_NATIVE_ENGINE = False
```

### Puertos

**Backend** (`src/config.py`):
```python
LAB_SERVER_HOST = '0.0.0.0'
LAB_SERVER_PORT = 8000
```

**Frontend** (`frontend/vite.config.ts`):
```typescript
server: {
  port: 5173,
  // ...
}
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'atheria_core'"

El motor nativo no está compilado. Compílalo:

```bash
cd /home/jonathan.correa/Projects/Atheria
source ath_venv/bin/activate
pip install -e .
```

### Error: "Port 8000 already in use"

Cambia el puerto del servidor en `src/config.py`:

```python
LAB_SERVER_PORT = 8001  # O cualquier otro puerto disponible
```

### Error: "Port 5173 already in use"

Cambia el puerto del frontend en `frontend/vite.config.ts`:

```typescript
server: {
  port: 5174,  // O cualquier otro puerto disponible
  // ...
}
```

### El frontend no se conecta al backend

1. Verifica que el backend esté corriendo en `http://localhost:8000`
2. Verifica que la URL del WebSocket en el frontend sea correcta
3. Revisa la consola del navegador para errores de conexión

## Uso con Motor Nativo

Cuando cargas un experimento:

1. El backend **automáticamente** busca el modelo JIT (`.pt`)
2. Si no existe, lo **exporta automáticamente** desde el checkpoint (`.pth`)
3. Carga el motor nativo con el modelo JIT
4. **¡Disfruta de 250-400x más rendimiento!**

### Verificación

En los logs del servidor, deberías ver:

```
✅ Motor nativo (C++) cargado exitosamente con modelo JIT
⚡ Motor nativo cargado (250-400x más rápido)
```

## Workflow Completo

1. **Iniciar Backend**:
   ```bash
   cd /home/jonathan.correa/Projects/Atheria
   source ath_venv/bin/activate
   python run_server.py
   ```

2. **Iniciar Frontend** (en otra terminal):
   ```bash
   cd /home/jonathan.correa/Projects/Atheria/frontend
   npm run dev
   ```

3. **Abrir Navegador**: `http://localhost:5173`

4. **Cargar Experimento**: Selecciona un experimento y presiona "Cargar"

5. **Iniciar Simulación**: Presiona "Iniciar" para comenzar la simulación

6. **¡Listo!**: La simulación debería ejecutarse **250-400x más rápido** si usas el motor nativo.

