# 2025-12-04: EXP-004: IonQ Engine Simulations

**Fecha:** 2025-12-04
**Tipo:** Experiment
**ID:** EXP-004

## Objetivo

Implementar scripts de simulación cuántica para los 5 motores principales en IonQ/Qiskit.

## Implementación

Creados 5 scripts en `scripts/` para diferentes tipos de evolución:
- Time Crystal Engine
- Harmonic Engine
- Lattice Engine
- Sparse Engine
- Polar Engine

Fallback robusto a `AerSimulator` cuando IonQ no está disponible.

## Resultados

- **Lattice:** Preservación de simetría de gauge confirmada
- **Sparse:** Disipación exitosa (~75.5% supervivencia)
- **Time:** Evolución con mayor dispersión (ruido simulado)

## Estado

✅ Completado (IonQ Simulator)

## Referencias

- [[EXP_005_HYBRID_HARMONIC]]
- [[ROADMAP_PHASE_4]]
