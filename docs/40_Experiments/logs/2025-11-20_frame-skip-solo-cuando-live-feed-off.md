## 2025-11-20 - Frame Skip Solo Cuando Live Feed OFF

### Contexto
Corrección para que `frame_skip` solo se aplique cuando `live_feed` está OFF.

### Problema Resuelto

#### Frame Skip Interfiriendo con Live Feed
- **Antes:** `frame_skip` se aplicaba siempre, incluso cuando `live_feed` estaba ON, causando frames saltados
- **Después:** `frame_skip` solo se aplica cuando `live_feed` está OFF

### Implementación

**Archivo:** `src/pipelines/pipeline_server.py`

**Cambios:**
- Verificar `live_feed_enabled` antes de aplicar `frame_skip`
- Si `live_feed` está ON, siempre enviar frames (no saltar)

### Estado
✅ **Completado**

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
