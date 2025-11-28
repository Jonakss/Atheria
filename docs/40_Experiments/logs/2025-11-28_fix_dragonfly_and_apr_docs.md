# 2025-11-28 - Fix: Dragonfly Startup & APR Documentation

**Contexto:**
- `docker-compose` fallaba debido a problemas con el entorno de Python del sistema (`http+docker` scheme not supported).
- Necesidad de documentar el sistema APR (Application Performance Repository) para modelos.

**Cambios:**
- **Dragonfly Fix:** Se optó por ejecutar Dragonfly directamente con `docker run` en lugar de `docker-compose` para evitar conflictos de dependencias de Python.
  - Comando: `docker run -d -p 6379:6379 --name atheria_dragonfly ...`
- **APR Documentation:** Se creó `docs/30_Components/APR_MODELS.md` definiendo métricas, benchmarks y protocolos para el análisis de modelos.
- **Tests:** Se verificó la conectividad con Dragonfly usando `tests/test_dragonfly_cache.py`.

**Archivos Relacionados:**
- [[APR_MODELS]]
- `tests/test_dragonfly_cache.py`
