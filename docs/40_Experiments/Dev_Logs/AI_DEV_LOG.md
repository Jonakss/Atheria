# 📝 AI Dev Log - Atheria 4

**Última actualización:** 2025-01-21  

**IMPORTANTE - Knowledge Base:** Este archivo es parte de la **BASE DE CONOCIMIENTOS** del proyecto. No es solo un log, es conocimiento que los agentes consultan para entender el contexto histórico y las decisiones tomadas. Ver [[00_KNOWLEDGE_BASE.md]] para más información.

**Objetivo:** Documentar decisiones de desarrollo, experimentos y cambios importantes para RAG y Obsidian.

**Reglas de actualización:**
- Actualizar después de cada cambio significativo o experimento
- Explicar **POR QUÉ** se tomó una decisión, no solo **QUÉ** se hizo
- Incluir referencias a código relacionado y otros documentos en `docs/`
- Usar enlaces `[[archivo]]` para conectar conceptos relacionados (formato Obsidian)

---

## 📋 Índice de Entradas

- [[logs/2025-11-26_fix_live_feed_logic_and_imports|2025-11-26 - Fix Lógica Live Feed y Errores de Importación]]
- [[logs/2025-11-26_fix_native_freeze_and_grid_scaling|2025-11-26 - Fix: Native Engine Freeze & Grid Scaling]]
- [[logs/2025-11-26_fullspeed_websocket_fix|2025-11-26 - Fix Saturación WebSocket en Modo Full Speed]]
- [[logs/2025-11-26_build_fixes_and_cli|2025-11-26 - Fix Crítico de Build y Mejoras CLI]]
- [[logs/2025-11-26_native_optimization_and_fixes|2025-11-26 - Optimización Crítica Motor Nativo (<1ms) y Fix Live Feed]]
- [[logs/2025-11-25_agent_config_sync|2025-11-25 - Sincronización de Configuración de Agentes (Lightning, Cursor, Gemini)]]
- [[logs/2025-11-25_native_engine_optimization|2025-11-25 - Native Engine Optimization & Fixes]]
- [[logs/2025-11-25_finalización-fase-1-y-verificación-motor-nativo|2025-11-25 - Finalización Fase 1 y Verificación Motor Nativo]]
- [[logs/2025-11-24_correcciones-ui-y-rendimiento-zoom-fps-throttling-y-native-engine|2025-11-24 - Correcciones UI y Rendimiento: Zoom, FPS, Throttling y Native Engine]]
- [[logs/2025-11-23_optimizaciones-críticas-de-live-feed-y-rendimiento|2025-11-23 - Optimizaciones Críticas de Live Feed y Rendimiento]]
- [[logs/2025-11-21_fix-carga-de-modelos-en-servidor-de-inferencia|2025-11-21 - Fix: Carga de Modelos en Servidor de Inferencia]]
- [[logs/2025-11-21_fix-configuración-de-proxy-websocket-en-frontend|2025-11-21 - Fix: Configuración de Proxy WebSocket en Frontend]]
- [[logs/2025-11-21_fase-2-paralelización-con-openmp-en-motor-nativo|2025-11-21 - Fase 2: Paralelización con OpenMP en Motor Nativo]]
- [[logs/2025-11-21_corrección-crítica-filtrado-de-propagación-z-en-motor-nativo|2025-11-21 - Corrección Crítica: Filtrado de Propagación Z en Motor Nativo]]
- [[logs/2025-01-21_corrección-fundamental-generación-de-estado-inicial-según-ley-m|2025-01-21 - Corrección Fundamental: Generación de Estado Inicial según Ley M]]
- [[logs/2025-01-XX_refactorización-progresiva-handlers-y-visualizaciones|2025-01-XX - Refactorización Progresiva: Handlers y Visualizaciones]]
- [[logs/2025-01-XX_documentación-análisis-atlas-del-universo|2025-01-XX - Documentación: Análisis Atlas del Universo]]
- [[logs/2025-01-XX_corrección-visualización-en-gris-normalización-de-map_data|2025-01-XX - Corrección: Visualización en Gris (Normalización de map_data)]]
- [[logs/2025-01-XX_sistema-de-versionado-automático-con-github-actions|2025-01-XX - Sistema de Versionado Automático con GitHub Actions]]
- [[logs/2025-01-XX_visualizaciones-con-shaders-webgl-gpu-implementadas|2025-01-XX - Visualizaciones con Shaders WebGL (GPU) Implementadas]]
- [[logs/2024-11-21_manejo-robusto-de-cuda-out-of-memory|2024-11-21 - Manejo Robusto de CUDA Out of Memory]]
- [[logs/2025-11-20_modo-manual-de-visualización-steps_interval-0|2025-11-20 - Modo Manual de Visualización (steps_interval = 0)]]
- [[logs/2025-11-20_separación-live-feed-binario-messagepack-vs-json|2025-11-20 - Separación Live Feed: Binario (MessagePack) vs JSON]]
- [[logs/2025-11-20_refactorización-archivos-atómicos-en-progreso|2025-11-20 - Refactorización: Archivos Atómicos (En Progreso)]]
- [[logs/2025-11-20_cli-simple-y-manejo-de-errores-robusto|2025-11-20 - CLI Simple y Manejo de Errores Robusto]]
- [[logs/2025-11-20_checkpoint-step-tracking-y-grid-scaling-info|2025-11-20 - Checkpoint Step Tracking y Grid Scaling Info]]
- [[logs/2025-11-20_frame-skip-solo-cuando-live-feed-off|2025-11-20 - Frame Skip Solo Cuando Live Feed OFF]]
- [[logs/2024-12-20_optimizaciones-críticas-motor-nativo-implementadas|2024-12-20 - Optimizaciones Críticas Motor Nativo Implementadas]]
- [[logs/2024-12-20_problemas-críticos-motor-nativo-identificados|2024-12-20 - Problemas Críticos Motor Nativo Identificados]]
- [[logs/2024-12-20_corrección-segfault-cleanup-motor-nativo|2024-12-20 - Corrección Segfault: Cleanup Motor Nativo]]
- [[logs/2024-12-XX_optimización-de-logs-y-reducción-de-verbosidad|2024-12-XX - Optimización de Logs y Reducción de Verbosidad]]
- [[logs/2024-12-XX_fase-3-completada-migración-de-componentes-ui|2024-12-XX - Fase 3 Completada: Migración de Componentes UI]]
- [[logs/2024-12-XX_fase-2-iniciada-setup-motor-nativo-c|2024-12-XX - Fase 2 Iniciada: Setup Motor Nativo C++]]
- [[logs/2025-01-21_mejoras-de-responsividad-y-limpieza-de-motor-nativo|2025-01-21 - Mejoras de Responsividad y Limpieza de Motor Nativo]]
