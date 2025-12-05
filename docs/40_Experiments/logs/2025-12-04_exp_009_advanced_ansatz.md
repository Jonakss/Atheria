# 2025-12-04: EXP-009: Advanced Ansatz (Strongly Entangling)

**Fecha:** 2025-12-04
**Tipo:** Experiment
**ID:** EXP-009

## Objetivo

Superar la baja fidelidad (1%) de EXP-008 usando un ansatz más expresivo.

## Implementación

- **Script:** `scripts/experiment_advanced_ansatz.py`
- **Ansatz:** "Strongly Entangling Layers" (Schuld et al.)
  - Rotaciones $U3(\theta, \phi, \lambda)$ en cada qubit (3 parámetros libres vs 1 en EXP-008)
  - Entrelazamiento CNOT circular (Topología de anillo)

## Resultados

- **Fidelidad:** **99.99%** (vs 1% en EXP-008)
- **Conclusión:** La falta de expresividad era el cuello de botella. Con un ansatz rico (U3 + Circular), podemos comprimir perfectamente el operador de evolución temporal.
- **Gate Count:** Lineal $O(N)$, viable para IonQ.

## Verificación IonQ

- Job ID: `019aeae2-9fd2-70d4-a72c-515f9682cc1f` (ionq_simulator)

## Estado

✅ Completado (Éxito Rotundo)

## Referencias

- [[EXP_008_QUANTUM_NATIVE_TRAINING]]
- [[ROADMAP_PHASE_4]]
