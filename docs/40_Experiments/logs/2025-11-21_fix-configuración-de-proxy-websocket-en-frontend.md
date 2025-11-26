## 2025-11-21 - Fix: Configuración de Proxy WebSocket en Frontend

### Problema
El frontend en desarrollo (`ath frontend-dev`) no podía conectarse al backend.

### Solución
Agregado proxy en `frontend/vite.config.ts`:
```typescript
server: {
  port: 3000,
  proxy: {
    '/ws': {
      target: 'ws://localhost:8000',
      ws: true,
      changeOrigin: true,
    },
  },
}
```

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
