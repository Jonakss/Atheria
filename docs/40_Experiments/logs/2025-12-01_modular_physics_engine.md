# 2025-12-01 - Modular Physics Engine Selection

## Summary
Implemented a modular architecture to select the Physics Engine (Cartesian, Polar, Quantum) when creating an experiment. This ensures the correct simulation backend is used and persisted for future sessions.

## Changes
- **Backend**:
    - Created `src/motor_factory.py` to centralize engine creation logic.
    - Updated `src/server/server_handlers.py` to use the factory and persist `ENGINE_TYPE` in `config.json`.
    - Updated `src/engines/qca_engine_polar.py` to be compatible with the system interface.
- **Frontend**:
    - Updated `LabSider.tsx` to include a "Motor Físico" selector as the first step in experiment creation.
- **Documentation**:
    - Created [[MOTOR_FACTORY_ARCHITECTURE]] to document the new architecture.

## Technical Details
- **Factory Pattern**: Decouples engine creation from server logic, allowing easy addition of new engines (Lattice, etc.).
