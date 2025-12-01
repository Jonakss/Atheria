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

- [[logs/2025-12-01_fix_fps_display|2025-12-01 - Fix: FPS Display in SimulationService]]
- [[UI_DASHBOARD_IMPROVEMENTS_2025_11_29|2025-11-29 - Feature: Scientific Metrics Display & UI Investigation]]
- **2025-11-28 - Fix: Docker Compose Compatibility & CUDA Optimization Backlog**
- [[logs/2025-11-28_holographic_viewer_and_engine_switching|2025-11-28 - Feature: Holographic Viewer & Engine Switching Docs]]
- [[logs/2025-11-27_fix_import_error_and_conflicts|2025-11-27 - Fix: Import Error in Native Engine Wrapper & Git Conflicts]]
- [[logs/2025-11-27_fix_ci_pip_cache_and_versioning|2025-11-27 - Fix: CI Pip Cache & Frontend Versioning Strategy]]
- [[logs/2025-11-27_Fixing_Notebook_and_Logger|2025-11-27 - Fix: Notebook Bugs, ExperimentLogger Paths & Agent Safety]]
- [[logs/2025-11-27_Native_Engine_Releases|2025-11-27 - System: Multi-Platform Native Engine Releases]]
- [[logs/2025-11-27_Notebook_Upgrade|2025-11-27 - Tool: Upgrade Notebook para Entrenamiento Progresivo Multi-Fase]]
- [[logs/2025-11-27_Progressive_Training_Notebook_Creation|2025-11-27 - Tool: Progressive Training Notebook (Long-Running GPU Sessions)]]
- [[logs/2025-11-26_advanced_field_visualizations|2025-11-26 - Feature: Advanced Field Visualizations (Real/Imag/HSV Phase)]]
- [[logs/2025-11-26_history_buffer_system|2025-11-26 - Feature: History Buffer System (Rewind/Replay)]]
- [[logs/2025-11-26_debugging_grid_canvas_versioning|2025-11-26 - Fix: Debugging Grid, Canvas, Versioning]]
- [[logs/2025-11-26_native_parallelism_openmp|2025-11-26 - Feature: Native Engine Parallelism (OpenMP)]]
- [[logs/2025-11-26_fix_persistent_frame_sending|2025-11-26 - Fix: Persistent Frame Sending (Duplicate Logic Removal)]]
- [[logs/2025-11-26_native_optimization_and_fixes|2025-11-26 - Optimización Crítica Motor Nativo (<1ms) y Fix Live Feed]]
- [[logs/2025-11-26_roadmap_updates|2025-11-26 - Actualización Completa de Roadmaps (Fases 1-4)]]
- [[logs/2025-11-26_fullspeed_websocket_fix|2025-11-26 - Fix Saturación WebSocket en Modo Full Speed]]
- [[logs/2025-11-26_fix_import_path_epoch_detector|2025-11-26 - Fix: Import Path de EpochDetector]]
- [[logs/2025-11-25_phase1_completion_native_verification|2025-11-25 - Finalización Fase 1 y Verificación Motor Nativo]]
- [[logs/2025-11-24_ui_performance_fixes|2025-11-24 - Correcciones UI y Rendimiento: Zoom, FPS, Throttling y Native Engine]]
- [[logs/2025-11-24_crash_loop_backend_fix|2025-11-24 - CRÍTICO: Solución Crash Loop Backend por Conversión Bloqueante]]
- [[logs/2025-11-23_critical_live_feed_optimizations|2025-11-23 - Optimizaciones Críticas de Live Feed y Rendimiento]]
- [[logs/2025-11-23_refactor_architecture_decoupled_services|2025-11-23 - Refactorización de Arquitectura: Servicios Desacoplados]]
- [[logs/2025-11-28_fix_dragonfly_and_apr_docs|2025-11-28 - Fix: Dragonfly Startup & APR Documentation]]
- [[logs/2025-11-28_dragonfly_cache_integration|2025-11-28 - Feature: Dragonfly Cache Integration (States & Checkpoints)]]
- [[logs/2025-11-28_fix_inference_persistence|2025-11-28 - Fix: Inference Persistence & Import Errors]]
- [[logs/2025-11-28_harlow_limit_theory|2025-11-28 - Knowledge: Harlow Limit Theory Ingestion]]
- [[logs/2025-11-28_fix_native_freeze|2025-11-28 - Fix: Native Engine Freeze (Fast Path Visualization)]]
- 2025-11-28: [PHASE 4] Initialization of Phase 4: Holographic Lattice.
    - Implemented `LatticeEngine` backend (Python prototype).
    - Integrated `LatticeEngine` into `inference_handlers.py`.
    - Created documentation `docs/30_Components/LATTICE_ENGINE.md`.
    - Planned `HolographicViewer2` for AdS/CFT visualization.
    - Implemented `HolographicViewer2` frontend component with visual indicator.
    - Added UI toggle in `PhysicsInspector` to switch between Viewer v1 and v2.
    - Updated `DashboardLayout` to handle viewer switching.
    - **Backend:** Implemented SU(3) Wilson Action and Metropolis-Hastings update in `LatticeEngine`.
    - **Frontend:** Implemented Scale-Radius Duality visualization using custom Vertex Shader in `HolographicViewer2`.
    - Verified backend physics with `tests/test_lattice_engine.py`.
- [[logs/2025-11-28_history_system_verification|2025-11-28 - Fix: History System Verification (Native Engine Support)]]
- [2025-11-28: Visualizaciones Avanzadas de Campos (WebGL)](logs/2025-11-28_advanced_visualizations.md)
- [2025-11-28: Implementación de OctreeIndex (Morton Codes)](logs/2025-11-28_octree_implementation.md)
- [2025-11-28: Optimización de Motor Nativo con OpenMP](logs/2025-11-28_openmp_optimization.md)
- [2025-11-28: Implementación de Proyección de Poincaré y Optimización de Quadtree](logs/2025-11-28_poincare_quadtree.md)
- [2025-11-28 - AdS/CFT Correspondence Documentation](logs/2025-11-28_ads_cft_correspondence.md)
- [[logs/2025-11-28_ads_cft_correspondence|2025-11-28 - Knowledge: AdS/CFT Correspondence & Documentation Fixes]]
- [2025-11-28: Integración de EpochDetector y Finalización de Fase 1](logs/2025-11-28_epoch_detector_integration.md)
- [2025-11-28: Reestructuración de Roadmap (AdS/CFT)](logs/2025-11-28_roadmap_restructuring_ads_cft.md)
- [[logs/2025-11-28_roadmap_restructuring_ads_cft|2025-11-28 - Strategy: Roadmap Restructuring (AdS/CFT, Infra, Research)]]
- [[#2025-01-21 - Corrección Fundamental: Generación de Estado Inicial según Ley M]]
- [[#2025-01-21 - Mejoras de Responsividad y Limpieza de Motor Nativo]]
- [[#2025-01-XX - Refactorización Progresiva: Handlers y Visualizaciones]]
- [[#2025-01-XX - Documentación: Análisis Atlas del Universo]]
- [[#2025-01-XX - Corrección: Visualización en Gris (Normalización de map_data)]]
- [[#2025-01-XX - Sistema de Versionado Automático con GitHub Actions]]
- [[#2025-01-XX - Visualizaciones con Shaders WebGL (GPU) Implementadas]]
- [[#2024-11-21 - Manejo Robusto de CUDA Out of Memory]]
- [[#2025-11-20 - Modo Manual de Visualización (steps_interval = 0)]]
- [[#2025-11-20 - Refactorización: Archivos Atómicos (En Progreso)]]
- [[#2025-11-20 - CLI Simple y Manejo de Errores Robusto]]
- [[#2025-11-20 - Checkpoint Step Tracking y Grid Scaling Info]]
- [[#2025-11-20 - Frame Skip Solo Cuando Live Feed OFF]]
- [[#2025-11-20 - Optimizaciones Críticas Motor Nativo Implementadas]]
- [[#2024-12-20 - Problemas Críticos Motor Nativo Identificados]]
- [[#2024-12-20 - Corrección Segfault: Cleanup Motor Nativo]]
- [[#2024-12-XX - Fase 3 Completada: Migración de Componentes UI]]
- [[#2024-12-XX - Fase 2 Iniciada: Setup Motor Nativo C++]]
- [[#2024-12-XX - Optimización de Logs y Reducción de Verbosidad]]
