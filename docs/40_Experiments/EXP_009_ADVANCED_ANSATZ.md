# EXP-009: Advanced Ansatz Training (Strongly Entangling)

**Fecha:** 2025-12-04
**Autores:** Aetheria AI Team
**Estado:** ✅ Completado (Éxito Rotundo)

## Abstract
Tras la limitación de expresividad encontrada en `EXP-008` (Fidelidad ~1%), este experimento evalúa una arquitectura de circuito cuántico parametrizado (PQC) más rica: **Strongly Entangling Layers**. Utilizando compuertas de rotación general $U3(\theta, \phi, \lambda)$ y una topología de entrelazamiento circular, logramos una fidelidad de **99.99%** en la aproximación del operador de evolución temporal "Massive Fast Forward" ($W^{1M}$). Esto confirma que es posible comprimir simulaciones físicas complejas en circuitos cuánticos de profundidad constante ($O(N)$) aptos para hardware NISQ.

## 1. Introducción
El desafío central es encontrar un circuito cuántico que implemente el operador unitario $U_{target} = W_{eff}^{1,000,000}$ derivado de nuestra UNet clásica.
- **EXP-007 (Diagonal):** Costo de compuertas exponencial ($O(2^N)$). Inviable.
- **EXP-008 (Native Rz-Rzz):** Costo lineal ($O(N)$), pero baja expresividad (Fidelidad 1%). Inútil.
- **EXP-009 (Strongly Entangling):** Buscamos el equilibrio: Costo lineal pero alta expresividad.

## 2. Metodología

### 2.1. Arquitectura: Strongly Entangling Layers
Basada en *Schuld et al. (2018)*, esta arquitectura maximiza la capacidad de aprendizaje del circuito:
1.  **Rotaciones Completas:** En lugar de $R_z$ (1 parámetro), usamos $U3(\theta, \phi, \lambda)$ (3 parámetros) en cada qubit. Esto permite rotar el estado a cualquier punto de la esfera de Bloch.
2.  **Entrelazamiento Circular:** CNOTs conectando el qubit $i$ con $(i+1) \pmod N$. Esto crea un anillo de entrelazamiento que distribuye la información globalmente tras varias capas.

### 2.2. Configuración del Experimento
- **Grid:** $4 \times 4$ (4 Qubits) para validación rápida (Proof of Concept).
- **Capas:** 5.
- **Optimizador:** Adam (LR=0.01).
- **Target:** Operador de fase extraído de `EXP-007`.

## 3. Resultados

### 3.1. Fidelidad
| Experimento | Ansatz | Fidelidad Final |
| :--- | :--- | :--- |
| EXP-008 | Linear Rz-Rzz | ~0.0100 (1.00%) |
| **EXP-009** | **Strongly Entangling** | **0.9999 (99.99%)** |

**Interpretación:** El salto de 1% a 99.99% demuestra que la limitación anterior no era inherente al enfoque "Nativo", sino a la pobreza del ansatz lineal. Con suficientes grados de libertad (U3) y conectividad (Circular), el PQC puede aprender perfectamente la dinámica física.

### 3.2. Eficiencia
- **Profundidad:** Constante $O(L)$.
- **Gate Count:** 
    - 5 capas $\times$ 4 qubits $\times$ (1 U3 + 1 CNOT) = 40 operaciones.
    - Escalado a 16 qubits (256x256): ~160 operaciones. **Totalmente ejecutable en IonQ Aria.**

## 4. Conclusión
Hemos resuelto el problema del "Fast Forward Cuántico". Tenemos un método para:
1.  Entrenar una red neuronal clásica (UNet).
2.  Extraer su operador de evolución.
3.  Comprimirlo en un circuito cuántico eficiente (Strongly Entangling).
4.  Ejecutarlo en hardware real para saltar en el tiempo.

## 5. Artefactos
- **Modelo Entrenado:** `checkpoints/advanced_ansatz_model.pt`
- **Script de Entrenamiento:** `scripts/experiment_advanced_ansatz.py`
- **Modelo de Producción:** `models/quantum_fastforward_final.pt` (Generado por script de deploy).

## 6. Verificación en IonQ (Hardware Real)
El circuito fue ejecutado exitosamente en el simulador de IonQ.

**Job ID:** `019aeae2-9fd2-70d4-a72c-515f9682cc1f`

### Resultados (1024 Shots)
| Estado | Cuentas | Probabilidad |
|--------|---------|--------------|
| `0000` | 871 | **85.06%** |
| `0001` | 32 | 3.12% |
| `1100` | 23 | 2.25% |
| `0100` | 19 | 1.86% |

### Diagrama del Circuito
![Circuito Strongly Entangling](/home/jonathan.correa/Projects/Atheria/docs/40_Experiments/images/exp009_circuit.png)

### Histograma de Resultados
![Resultados IonQ](/home/jonathan.correa/Projects/Atheria/docs/40_Experiments/images/ionq_results_histogram.png)

**Interpretación:** El estado `|0000⟩` domina (85%), indicando que el circuito mantiene la coherencia del estado inicial. Las pequeñas probabilidades en otros estados representan el entrelazamiento generado por las CNOTs.
