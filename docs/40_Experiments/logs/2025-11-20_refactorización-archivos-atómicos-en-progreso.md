## 2025-11-20 - Refactorización: Archivos Atómicos (En Progreso)

### Contexto
El archivo `pipeline_server.py` tenía 3,567 líneas con 37 handlers, lo que hacía difícil mantener el código, buscar funcionalidades específicas y reducir el contexto necesario en los chats de IA.

### Objetivo
Factorizar `pipeline_server.py` en módulos más pequeños y atómicos (~300-700 líneas cada uno) para:
- Reducir contexto necesario en chats (de 3,567 → ~300-700 líneas por módulo)
- Facilitar búsquedas precisas
- Mejorar mantenibilidad y testing
- Reducir conflictos en colaboración

### Estructura Propuesta

```
src/pipelines/
├── server.py                    # Archivo principal (reducido ~500 líneas)
├── handlers/                    # Módulos de handlers (~300-700 líneas cada uno)
│   ├── experiment_handlers.py   ✅ CREADO
│   ├── simulation_handlers.py   ⏳ PENDIENTE
│   ├── inference_handlers.py    ⏳ PENDIENTE
│   ├── analysis_handlers.py     ⏳ PENDIENTE
│   ├── visualization_handlers.py ⏳ PENDIENTE
│   ├── config_handlers.py       ⏳ PENDIENTE
│   └── system_handlers.py       ⏳ PENDIENTE
├── core/                        # Componentes core
│   ├── websocket_handler.py     ⏳ PENDIENTE
│   ├── simulation_loop.py       ⏳ PENDIENTE
│   └── route_setup.py           ⏳ PENDIENTE
└── viz/                         # Visualizaciones
    ├── basic.py                 ⏳ PENDIENTE
    ├── advanced.py              ⏳ PENDIENTE
    └── physics.py               ⏳ PENDIENTE
```

### Progreso

#### ✅ Completado
1. **Plan de Refactorización**: Documentado en `docs/30_Components/REFACTORING_PLAN.md`
2. **Estructura de Directorios**: Creados `handlers/`, `core/`, y `viz/`
3. **experiment_handlers.py**: Módulo creado con handlers de experimentos:
   - `handle_create_experiment()`
   - `handle_continue_experiment()`
   - `handle_stop_training()`
   - `handle_delete_experiment()`
   - `handle_list_checkpoints()`
   - `handle_delete_checkpoint()`
   - `handle_cleanup_checkpoints()`
   - `handle_refresh_experiments()`

#### ⏳ Pendiente
1. Crear módulos restantes de handlers (simulation, inference, analysis, visualization, config, system)
2. Extraer `websocket_handler()` y `simulation_loop()` a módulos core
3. Refactorizar `pipeline_viz.py` en módulos de visualización
4. Actualizar `pipeline_server.py` para usar los nuevos módulos
5. Actualizar imports en otros archivos que usen handlers

### Beneficios Esperados

1. **Contexto Reducido**: De 3,567 líneas → ~300-700 líneas por módulo
2. **Búsquedas Más Precisas**: Buscar en módulo específico en lugar de archivo grande
3. **Mantenibilidad**: Cambios aislados en módulos específicos
4. **Testing**: Tests unitarios más fáciles por módulo
5. **Colaboración**: Menos conflictos, cambios más aislados

### Referencias
- [[REFACTORING_PLAN]] - Plan completo de refactorización
- `src/pipelines/handlers/experiment_handlers.py` - Módulo de handlers de experimentos

---



---
[[AI_DEV_LOG|🔙 Volver al Índice]]
