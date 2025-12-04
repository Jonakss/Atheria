# EXP-007: Massive Fast Forward & Closed Loop

**Fecha:** 2025-12-04
**Estado:** ✅ Completado
**Scripts:** 
- `scripts/experiment_massive_fastforward.py` (Simulación Local)
- `scripts/experiment_massive_fastforward_ionq.py` (IonQ Hardware/Simulator)

## 1. Objetivo
Demostrar la capacidad de realizar un "salto temporal masivo" (Massive Fast Forward) de 1 millón de pasos de simulación en una sola operación, utilizando la **Capa Neuronal Holográfica**. Además, verificar la viabilidad de ejecutar este proceso en hardware cuántico (IonQ) y cerrar el ciclo de inferencia (Quantum -> Classical).

## 2. Fundamento Teórico
Si la dinámica de un sistema es unitaria ($U = e^{-iHt}$), la evolución por $N$ pasos es $U^N$.
En una red neuronal estándar, calcular $U^N$ requiere $N$ pasadas forward ($O(N)$).
Sin embargo, si podemos diagonalizar el operador en el dominio de la frecuencia (Holografía):
$$ U = \mathcal{F}^{-1} \cdot D \cdot \mathcal{F} $$
Entonces:
$$ U^N = \mathcal{F}^{-1} \cdot D^N \cdot \mathcal{F} $$
Donde $D^N$ es simplemente potenciar los elementos de la diagonal (fases), lo cual es una operación $O(1)$ independiente de $N$.

## 3. Metodología

### A. Extracción del Operador ($W_{eff}$)
1.  Se cargó un modelo `UNetUnitary` pre-entrenado (Checkpoint: `UNetUnitary_G64_Eps130`).
2.  Se pasó un estado de prueba ("probe state") a través del modelo para caracterizar su respuesta en frecuencia.
3.  Se calculó la función de transferencia efectiva $H(k) = Y(k) / X(k)$.

### B. Fast Forward (Potenciación)
1.  Se calculó $H_{final} = H(k)^{1,000,000}$.
2.  Esto equivale a aplicar la evolución de 1 millón de pasos en un solo instante.

### C. Ejecución en IonQ (Closed Loop)
1.  **Compilación:** Se construyó un circuito cuántico con 10 qubits (Grid 32x32).
    - `H` (Inicialización)
    - `QFT`
    - `Diagonal(H_final)` (La "máscara holográfica")
    - `IQFT`
    - `Measure`
2.  **Ejecución:** Se envió el job al `ionq_simulator`.
3.  **Reconstrucción:** Se tomaron los *counts* resultantes y se reconstruyó la imagen (amplitud).
4.  **Continuación:** Se alimentó la imagen reconstruida de vuelta a la UNet en PyTorch para verificar que el ciclo es funcional.

## 4. Resultados

### Simulación Local (PyTorch/Qiskit Statevector)
- **Grid:** 64x64
- **Resultado:** Éxito total. Energía conservada.
- **Checkpoint:** `checkpoints/fastforward_1M.pt`

### Ejecución IonQ (Hardware Compatible)
- **Grid:** 32x32 (10 Qubits)
    - *Nota:* Se intentó 256x256 (16 Qubits) pero la descomposición de la compuerta diagonal excedió el límite de compuertas de la API (`TooManyGates`).
- **Job ID:** (Ver logs)
- **Resultado:** Éxito. Se recuperó el estado y se continuó la inferencia.
- **Checkpoint:** `checkpoints/fastforward_1M_ionq_loop.pt`

## 5. Conclusiones
- **Viabilidad Confirmada:** Es posible usar procesadores cuánticos como "aceleradores temporales" para simulaciones físicas gobernadas por redes neuronales unitarias.
- **Ventaja Cuántica:** La complejidad temporal pasa de $O(N)$ (clásico) a $O(1)$ (cuántico, una vez diagonalizado), con el costo fijo de QFT/IQFT ($O(\log^2 M)$).
- **Limitaciones Actuales:** El tamaño del grid en hardware real está limitado por la fidelidad y el tamaño del payload de la API para compuertas arbitrarias.

### Hallazgo Crítico: La Barrera de la Diagonal
Durante la ejecución en IonQ, descubrimos que implementar una compuerta `Diagonal` arbitraria (que representa nuestros pesos $W_{freq}$) tiene un costo exponencial en compuertas estándar (CNOTs).
- **10 Qubits (32x32):** Funciona (~1k CNOTs).
- **16 Qubits (256x256):** Falla (`TooManyGates`). Requiere $2^{16} = 65,536$ fases, lo que se descompone en una cantidad inmanejable de compuertas físicas.

**Solución Propuesta (EXP-008):**
En lugar de aprender una matriz arbitraria y tratar de descomponerla, debemos usar **Entrenamiento Cuántico Nativo**.
- Definir un circuito parametrizado (PQC) con profundidad fija $O(N)$ o $O(N \log N)$.
- Entrenar los ángulos de rotación de este circuito directamente.
- Esto garantiza que el operador resultante sea siempre eficiente de ejecutar en hardware.

## 6. Próximos Pasos
- **EXP-008:** Entrenar una `QuantumNativeConv2d` (basada en compuertas $R_z$ y $R_{zz}$) para aproximar el operador de Fast Forward, reduciendo el costo de $O(2^N)$ a $O(poly(N))$.
- Probar en QPU real (Harmony/Aria) con un grid pequeño (4x4 o 8x8).
