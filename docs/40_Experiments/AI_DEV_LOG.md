# 📝 AI Dev Log - Atheria 4

**Última actualización:** 2025-12-01

**IMPORTANTE - Knowledge Base:** Este archivo es parte de la **BASE DE CONOCIMIENTOS** del proyecto. No es solo un log, es conocimiento que los agentes consultan para entender el contexto histórico y las decisiones tomadas. Ver [[00_KNOWLEDGE_BASE.md]] para más información.

**Objetivo:** Documentar decisiones de desarrollo, experimentos y cambios importantes para RAG y Obsidian.

**Reglas de actualización:**
- Actualizar después de cada cambio significativo o experimento
- Explicar **POR QUÉ** se tomó una decisión, no solo **QUÉ** se hizo
- Incluir referencias a código relacionado y otros documentos en `docs/`
- Usar enlaces `[[archivo]]` para conectar conceptos relacionados (formato Obsidian)

---

## 📋 Índice de Entradas

- **[[logs/2025-12-01_engine_architecture_refactor|2025-12-01 - Refactor: Engine Architecture (Physics vs Backend)]]**:
  - **Backend Separation**: Moved `SparseQuantumEngineCpp/V2` to `src/engines/backends/` and renamed to `SparseBackendCpp/V2`. Removed trainer methods from backends.
  - **Engine Renaming**: Renamed `Aetheria_Motor` to `CartesianEngine` to clarify its role as the Standard QCA engine.
  - **Frontend Prep**: Created prompt for Jules to implement Engine/Backend selectors.
- **[[PHASE_STATUS_REPORT]]**: 2025-12-01 (17:28) - Actualización comprehensiva del estado del proyecto. Progreso de Fase 2 (benchmarking Python vs C++, bloqueo en Native Engine), Fase 3 100% completado (frontend linting, Phase Space Viz, Field Theory UI), y Compute Backend 100% completado. Tareas críticas actualizadas.
- **Fixing Engine Compatibility**:
  - **Native Engine Crash**: Mitigated by forcing `CARTESIAN` engine (Python backend) in `config.json` and `trainer.py` arguments.
  - **Lattice Engine**: Implemented missing methods (`compile_model`, `get_model_for_params`, `get_initial_state`, `evolve_step` aliased to `step`) to satisfy `QC_Trainer_v3` interface.
  - **Harmonic Engine**: Implemented missing methods (`compile_model`, `get_model_for_params`, `get_initial_state`, `evolve_step` aliased to `step`) to satisfy `QC_Trainer_v4` interface.
  - **Trainer Updates**:
    - Updated `QC_Trainer_v3` to handle engines without trainable models (skipping optimizer/scheduler init, gradient steps, and state dict saving).
    - Updated `QC_Trainer_v4` to handle engines without `state.psi` attribute (using `get_initial_state` fallback).
    - Added `CARTESIAN` to allowed `engine_type` choices in `trainer.py`.
  - **Verification**: Successfully verified training loop for `LATTICE`, `HARMONIC`, and `CARTESIAN` engines.
## [2025-12-02] Advanced Quantum Features Implementation
-   **Quantum Tuner**: Implementado `scripts/quantum_tuner.py` con optimización SPSA para encontrar estados iniciales complejos.
-   **Hybrid Compute**: Creado `src/physics/quantum_collapse.py` (`IonQCollapse`) para inyectar colapsos dependientes del estado durante la simulación.
-   **Quantum Steering**: Implementado `src/physics/steering.py` (`QuantumSteering`) y `handle_interaction` para permitir "Quantum Brush" (Vortex, Soliton) desde el frontend.
-   **Quantum Microscope**: Implementado `src/physics/quantum_kernel.py` (`QuantumMicroscope`) usando Deep Quantum Kernels (ZZ Feature Map) para análisis de complejidad estructural.
-   **Docs**: Creado `docs/20_Concepts/ADVANCED_QUANTUM_FEATURES.md` y actualizado `FRONTEND_JULES_PROMPT.md`.
-   **Verificación**: Ejecutado `scripts/run_full_quantum_experiment.py` con éxito (Tuner Score ~83, Steering Delta ~170).
-   **Frontend Repro**: Generado `docs/40_Experiments/frontend_repro_data.json` con payloads de ejemplo para WebSocket (`interaction_ack`, `quantum_analysis_result`).
-   **Lifecycle Demo**: Creado y ejecutado `scripts/experiment_quantum_lifecycle.py` demostrando el ciclo completo: Entrenamiento Híbrido -> Tuning Cuántico -> Inferencia Interactiva (Steering + Microscope).

## [2025-12-02] Quantum Multiverse Implementation
- **[[logs/2025-12-02_fix_polar_engine_and_trainer|2025-12-02 - Fix: Polar Engine Compatibility & Trainer Arguments]]**:
  - **Polar Engine Fix**: Added `.real`, `.imag`, and `.abs()` to `QuantumStatePolar` to satisfy `QC_Trainer_v4` interface. Optimized `evolve_step` to use `to_cartesian()` if available. Fixed device placement in `__init__` by adding `self.model.to(device)`.
  - **Harmonic Engine Fix**: Fixed dimension mismatch in `get_viewport_tensor` by changing output shape from `[1, C, H, W]` to `[1, H, W, C]` to match `CartesianEngine` and `VisualizationPipeline` expectations.
  - **Verification**: Verified with reproduction scripts `tests/reproduce_polar_error.py` and `tests/reproduce_harmonic_error.py`.
- **[[logs/2025-12-02_roi_visualization_improvements|2025-12-02 - Feature: ROI Visualization Improvements ("See All" Toggle & Gradient Overlay)]]**
- **2025-12-02**: Refactored Quantum Tools (`IonQCollapse`, `QuantumSteering`) into `src/physics/` and exposed them via `__init__.py`. Added unit tests `tests/test_quantum_tools.py`. Implemented Frontend Quantum Toolbox (`QuantumToolbox.tsx`) and integrated it into `LabSider`. Added backend handler `handle_tool_action` for real-time interaction. Fixed `train_progressive.py` root detection and `qc_trainer_v4.py` type hints. [Log](logs/2025-12-02_quantum_tools_refactor_and_frontend.md)
- **2025-12-02**: Implemented Cache Buffering (Streaming) using Dragonfly. Decoupled simulation speed from frontend visualization by pushing frames to a Redis list (`simulation:stream`) and consuming them at a constant rate. Added `CACHE_BUFFERING_ENABLED` to config. Updated `DataProcessingService` (producer) and `WebSocketService` (consumer). Verified with `scripts/verify_buffering.py`. [Log](logs/2025-12-02_cache_buffering_implementation.md)
- **2025-12-01**: Fixed `motor_factory.py` to support `LATTICE` and `HARMONIC` engines. Added `shape` property to `QuantumStatePolar`. Investigated Native Engine crash (`terminate called recursively`) but could not resolve it without C++ debugging access; suspected tensor interface mismatch. [Log](logs/2025-12-01_fix_engine_support_and_native_crash.md)
- [[logs/2025-12-01_fix_trainer_engine_and_motor_factory|2025-12-01 - Fix: Trainer Engine Type Support & Motor Factory Signature]]
- [[logs/2025-12-01_obsidian_kb_optimization|2025-12-01 - Optimization: Knowledge Base Obsidian Links (docs/20_Concepts/)]]
- [[logs/2025-12-01_frontend_lint_fixes|2025-12-01 - Fix: Frontend Lint and Build Errors]]
- [2025-12-01: UI Consolidation and Holographic Viewer Improvements](logs/2025-12-01_ui_consolidation.md)
- [[logs/2025-12-01_compute_backend_implementation|2025-12-01 - Feature: Compute Backend Abstraction (Local/Quantum/Mock)]]
- [[logs/2025-12-01_viz_ui_refactor_and_native_fixes|2025-12-01 - Refactor: Visualization UI & Native Engine Fixes]]
- [[logs/2025-12-01_engine_selection_training|2025-12-01 - Feature: Engine Selection for Training]]
- **2025-12-01 - Fix: Syntax Error in TransferLearningWizard (Extra Tag)**
- [[logs/2025-12-01_quantum_tuner|2025-12-01 - Feature: Quantum Tuner (Qiskit SPSA Optimization)]]
- [[logs/2025-12-01_field_theory_ui|2025-12-01 - Feature: Field Theory UI Implementation (Field Selector)]]
- [[logs/2025-12-01_fix_fps_display|2025-12-01 - Fix: FPS Display in SimulationService]]
- [[docs/40_Experiments/EXP_005_QUALITY_DIVERSITY|2025-12-01 - Research: Quantum Architecture & Polar Migration (VQE, MAP-Elites, PennyLane, Polar Engine)]]
- [[docs/10_core/ROADMAP_INFERENCE_OPTIMIZATION|2025-12-01 - Strategy: Inference Optimization Roadmap (LitServe, Quantization, Compile)]]
- [[logs/2025-12-01_fix_native_freeze|2025-12-01 - Fix: Native Engine JIT Export Freeze]]
- [[logs/2025-12-01_phase_space_viz|2025-12-01 - Feature: Phase Space Visualization (PCA + Clustering)]]
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
- **[[2025-12-01_benchmarking]]**: Resultados preliminares del benchmark comparativo. Python Engine alcanza ~60 FPS en CPU. Native Engine presenta bloqueos durante warmup que requieren debugging.
- **[[2025-12-01_native_performance]]**: Optimización crítica en `sparse_engine.cpp` reduciendo `patch_size` a 3x3, resolviendo el hang en benchmarks.
- **[[2025-12-01_memory_pools]]**: Implementación de `TensorPool` en C++ para reutilización de memoria y reducción de overhead de asignación.
- **[[2025-12-01_compute_backend]]**: Diseño de la arquitectura `ComputeBackend` para abstracción de hardware (CPU/GPU/QPU).
- **[[2025-12-01_octree_integration]]**: Integración de Octree en el motor nativo para optimización espacial 3D.
- **[[2025-12-01_modular_physics_engine]]**: Refactorización para selección modular de motores de física (Cartesiano, Polar, Cuántico).
- **[[2025-12-01_phase_space_viz]]**: Implementación de visualización de Espacio de Fases con PCA y UMAP.Clustering)]]
- [[#2025-11-20 - Optimizaciones Críticas Motor Nativo Implementadas]]
- [[#2024-12-20 - Problemas Críticos Motor Nativo Identificados]]
- [[#2024-12-20 - Corrección Segfault: Cleanup Motor Nativo]]
- [[#2024-12-XX - Fase 3 Completada: Migración de Componentes UI]]
- [[#2024-12-XX - Fase 2 Iniciada: Setup Motor Nativo C++]]
- [[#2024-12-XX - Optimización de Logs y Reducción de Verbosidad]]

- [[logs/2025-12-01_phase_space_viz|2025-12-01 - Feature: Phase Space Visualization (PCA + Clustering)]]
- [[logs/2025-12-01_modular_physics_engine|2025-12-01 - Feature: Modular Physics Engine Selection]]
- [[logs/2025-12-01_octree_integration|2025-12-01 - Feature: Octree Integration in Native Engine]]
- [[logs/2025-12-01_compute_backend|2025-12-01 - Feature: Compute Backend Abstraction]]
- [[logs/2025-12-01_memory_pools|2025-12-01 - Feature: Memory Pools & Concurrency Fixes]]
- [[logs/2025-12-01_native_performance|2025-12-01 - Performance: Native Engine Patch Size Optimization]]
- **[[logs/2025-12-02_frontend_build_and_lint_fixes|2025-12-02 - Fix: Frontend Build and Lint for Deployment]]**:
  - **Linting Fixes**: Resolved unused `useRef` and missing `useEffect` dependency in `HistoryControls.tsx`.
  - **Build Fixes**: Removed unused `@ts-expect-error` directive in `PhaseSpaceViewer.tsx` that was causing build failure.
  - **Verification**: Verified successful `npm run lint` and `npm run build`. Frontend is ready for deployment.
- **[[logs/2025-12-02_native_engine_crash_fix|2025-12-02 - Critical Fix: Native Engine Crash & Quantum Tools Verification]]**:
  - **Native Engine Fix**: Resolved `terminate called recursively` crash in `atheria_core`.
    - **Cause**: Uncaught exception in OpenMP parallel region during batch processing and thread-safety issue with CUDA generator in `HarmonicVacuum`.
    - **Resolution**: Added `try-catch` block in `sparse_engine.cpp` and switched `HarmonicVacuum` to use CPU-based noise generation (thread-safe) with explicit move to device.
  - **Epoch Detector Fix**: Resolved tensor shape mismatch (`(C, H, W)` vs `(H, W, C)`) in `EpochDetector` for `PolarEngine` and `HarmonicEngine`.
  - **Quantum Tools**: Verified integration of `IonQCollapse` and `QuantumSteering` with `NativeEngineWrapper` via `apply_tool`.
  - **Verification**: Verified fix with `reproduce_crash.py` on CUDA (64x64 and 512x512).
- **[[logs/2025-12-02_fix_polar_viz_error|2025-12-02 - Fix: Polar Engine Visualization Error]]**:
  - **Issue**: `TypeError: 'QuantumStatePolar' object is not subscriptable` in `calculate_poincare_coords` and `calculate_phase_attractor`.
  - **Fix**: Updated `src/pipelines/viz/advanced.py` to handle `QuantumStatePolar` objects (via `.to_cartesian()` or `.squeeze()`) and correctly permute Channels-First tensors `(C, H, W)` to `(H, W, C)` expected by visualization functions.
- **[[logs/2025-12-02_spectrum_viz_and_snapshot_fix|2025-12-02 - Fix: Spectrum Visualization & Snapshot Handler]]**:
  - **Spectrum Viz**: Fixed "single particle" appearance in spectral view by masking the DC component (zero frequency) which was dominating the dynamic range.
  - **Snapshot Fix**: Resolved `ValueError` when saving snapshots with `LatticeEngine` by adding a check for `get_dense_state` method in `snapshot_handlers.py`.
  - **Verification**: Verified spectrum improvement with `reproduce_spectrum.py` and confirmed snapshot handler logic via code analysis.
- **[[logs/2025-12-03_native_engine_deadlock_fix_and_benchmark|2025-12-03 - Critical Fix: Native Engine Deadlock & Benchmark Results]]**:
  - **Deadlock Fix**: Resolved Native Engine hang during warmup/initialization.
    - **Cause**: Deadlock caused by calling `torch::set_num_threads(1)` inside an OpenMP parallel region (`#pragma omp parallel`) in `sparse_engine.cpp`.
    - **Resolution**: Removed the problematic call. Verified fix with `scripts/test_native_quick.py` on CPU and CUDA.
  - **Benchmark Results**: Ran `scripts/benchmark_comparison.py` (32x32 grid).
    - **Python**: ~10.8 FPS.
    - **Native (C++)**: < 0.2 FPS (CPU).
    - **Analysis**: Native Engine is significantly slower due to massive overhead in `step_native` (batch construction, map access, small batch dispatch).
  - **Next Steps**: Optimization of memory management (pools) and batching strategy.
- **[[logs/2025-12-03_thread_local_tensor_pool|2025-12-03 - Optimization: Thread-Local Tensor Pools]]**:
  - **Feature**: Implemented `ThreadLocalTensorPool` in `src/cpp_core/include/tensor_pool.h`.
  - **Goal**: Remove mutex contention in `TensorPool` during OpenMP parallel execution.
  - **Implementation**: Replaced single `std::stack` with `std::vector<std::stack>` (one per thread) and removed `std::mutex`.
  - **Result**: Functional correctness verified. Performance impact on CPU (16x16 grid) is minimal, indicating bottleneck is elsewhere (batch overhead).
  - **Next**: Focus on `build_batch_input` optimization.
- **[[logs/2025-12-03_holographic_engine_implementation|2025-12-03 - Feature: Holographic Engine (AdS/CFT Projection)]]**:
  - **Concept**: Implemented `HolographicEngine` based on the Holographic Principle. It evolves a 2D boundary state but provides a 3D bulk projection via Scale-Space renormalization.
  - **Implementation**: Created `src/engines/holographic_engine.py` inheriting from `CartesianEngine`. Implemented `get_bulk_state()` using Gaussian blurring to simulate depth/scale.
  - **Integration**: Registered `HOLOGRAPHIC` engine type in `src/motor_factory.py`.
  - **Verification**: Verified initialization and projection logic (variance reduction with depth) via `tests/test_holographic_engine.py`.