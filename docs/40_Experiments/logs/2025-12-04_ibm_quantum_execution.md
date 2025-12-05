# 2025-12-04: Ejecución Multi-Plataforma (IonQ + IBM)

**Fecha:** 2025-12-04
**Tipo:** Feature

## Objetivo

Validar circuitos en hardware cuántico real para confirmar que el pipeline de compresión de operador temporal funciona correctamente en hardware real.

## Resultados

- **IonQ Simulator:** Job `019aeaf3-...` - Estado `|0000⟩` 85%.
- **IBM Fez (Real QPU):** Job `d4ouqhft3pms7395ki80` - Estado `|0000⟩` **90.6%**.
- **Tiempo Ejecución IBM:** ~5 segundos.

## Conclusión

El circuito variacional funciona correctamente en hardware real de ambas plataformas. La diferencia de fidelidad (85% IonQ vs 90.6% IBM) se debe a las características específicas de cada hardware.

## Scripts

- `scripts/run_ibm_now.py`
- `scripts/run_json_circuit_ionq.py`

## Estado

✅ Completado

## Referencias

- [[EXP_009_ADVANCED_ANSATZ]]
- [[ROADMAP_PHASE_4]]
