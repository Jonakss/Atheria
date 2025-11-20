# Configuración del Servidor Atheria

## 🚀 Iniciar el Servidor

### Modo Normal (con Frontend)

```bash
python3 run_server.py
```

El servidor intentará servir el frontend desde `frontend/dist/` si está disponible.

### Modo API Solo (sin Frontend)

```bash
python3 run_server.py --no-frontend
```

El servidor funcionará solo con WebSocket API. Útil cuando:
- El frontend se sirve desde otro servidor (Vite dev server, nginx, etc.)
- Solo se necesita la API WebSocket
- Desarrollo separado de frontend y backend

### Variables de Entorno

También puedes usar una variable de entorno:

```bash
# Desactivar frontend
ATHERIA_NO_FRONTEND=1 python3 run_server.py

# Activar frontend (por defecto)
python3 run_server.py
```

## 📡 Endpoints Disponibles

### Siempre Disponible

- **WebSocket API**: `/ws` - API principal para comunicación en tiempo real
- **Root**: `/` - Mensaje informativo o frontend (según configuración)

### Solo con Frontend Activado

- **Frontend SPA**: Todas las rutas sirven `index.html` para React Router
- **Archivos Estáticos**: CSS, JS, imágenes desde `frontend/dist/`

## ⚙️ Parámetros de Línea de Comandos

```bash
python3 run_server.py [OPTIONES]

Opciones:
  --no-frontend    No servir el frontend estático, solo WebSocket API
  --port PORT      Puerto del servidor (por defecto: 8000)
  --host HOST      Host del servidor (por defecto: 0.0.0.0)
  --help           Mostrar ayuda
```

## 🔧 Ejemplos de Uso

### Desarrollo Frontend Separado

```bash
# Terminal 1: Backend API solo
python3 run_server.py --no-frontend

# Terminal 2: Frontend con Vite dev server
cd frontend && npm run dev
```

### Producción con Frontend Build

```bash
# Build del frontend
cd frontend && npm run build

# Servidor completo (frontend + API)
python3 run_server.py
```

### Solo API Backend

```bash
# Servidor solo API (sin frontend)
python3 run_server.py --no-frontend

# Clientes se conectan a ws://localhost:8000/ws
```

## 📝 Notas

- El WebSocket API (`/ws`) siempre está disponible, independientemente de `--no-frontend`
- Si el frontend no está builded y no usas `--no-frontend`, verás un mensaje informativo en `/`
- El parámetro `--no-frontend` es útil para desarrollo separado o cuando el frontend se sirve desde otro servidor

---

**Última actualización:** 2024-11-20

