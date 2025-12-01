# 2025-12-01 - Feature: Quantum Tuner (Qiskit Integration)

## 🎯 Objetivo
Implementar un sistema de optimización de hiperparámetros (`GAMMA_DECAY`, `LR_RATE`) utilizando computación cuántica variacional (VQC) a través de Qiskit Runtime.

## 🛠️ Implementación
Se ha creado el script `scripts/quantum_tuner.py` que actúa como un "Orquestador Cuántico-Clásico".

### Componentes:
1.  **Circuito Variacional (The Explorer):**
    - 2 Qubits con puertas `RX`, `RY` y entrelazamiento `CX`.
    - Parámetros $\theta$ y $\phi$ controlan la exploración del espacio de búsqueda.
2.  **Mapeo de Parámetros:**
    - $\theta \to \text{GAMMA\_DECAY}$ (Rango: 0.0 - 0.15)
    - $\phi \to \text{LR\_RATE}$ (Rango: 0.0001 - 0.01)
3.  **Función de Costo (The Judge):**
    - Ejecuta una simulación corta de Aetheria (50 pasos).
    - Calcula la **Entropía** del estado final.
    - Objetivo: **Maximizar la Entropía** (buscamos complejidad).
4.  **Optimizador SPSA:**
    - Algoritmo ideal para entornos ruidosos (NISQ).
    - Optimiza los parámetros del circuito para minimizar la función de costo ($-1 \times \text{Entropía}$).

## 📦 Dependencias Nuevas
Se agregaron las siguientes librerías a `requirements.txt`:
- `qiskit`
- `qiskit-algorithms`
- `qiskit-ibm-runtime`

## 📝 Notas Técnicas
- El script detecta automáticamente si `qiskit-ibm-runtime` está disponible. Si no, usa `StatevectorEstimator` local.
- Se corrigió un problema de compatibilidad en `src/config.py` (falta de `pathlib`).
- Se mejoró la robustez del script para manejar valores `NaN` o `Inf` en la entropía, retornando una penalización alta para guiar al optimizador lejos de zonas inestables.

## 🚀 Uso
```bash
python3 scripts/quantum_tuner.py
```
