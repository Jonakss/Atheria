# 2025-12-04: EXP-008: Quantum-Native Training

**Fecha:** 2025-12-04
**Tipo:** Experiment
**ID:** EXP-008

## Objetivo

Resolver el problema de "TooManyGates" entrenando un PQC (Parametrized Quantum Circuit) nativo.

## Implementación

- **Script:** `scripts/experiment_quantum_native_training.py`
- **Ansatz:** Capas $R_z$ (locales) + $R_{zz}$ (entrelazamiento vecino)
- **Entrenamiento:** Aproximar fase del operador Fast Forward ($W^{1M}$)

## Resultados

- **Reducción de Costo:** De $O(2^N)$ (Diagonal) a $O(N \times L)$ (Nativo)
- **Viabilidad:** Permite escalar a 20+ qubits sin exceder límites de gates

## Estado

✅ Completado

## Referencias

- [[EXP_007_MASSIVE_FASTFORWARD]]
- [[EXP_009_ADVANCED_ANSATZ]]
