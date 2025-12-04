# Holographic Neural Layer

**Componente:** `HolographicConv2d`
**Ubicación:** `scripts/experiment_holographic_layer.py` (Prototipo) / `src/models/layers/holographic.py` (Futuro)
**Estado:** Experimental (Prototipo Funcional)

## Descripción
La **Capa Neuronal Holográfica** es un módulo de red neuronal que realiza operaciones de convolución utilizando principios de física cuántica y óptica holográfica. En lugar de realizar una convolución espacial estándar (producto punto deslizante), transforma la señal de entrada al dominio de la frecuencia, aplica una "máscara holográfica" (pesos complejos), y transforma de vuelta al dominio espacial.

## Fundamento Físico
Se basa en el **Teorema de Convolución**:
$$ f * g = \mathcal{F}^{-1} \{ \mathcal{F}\{f\} \cdot \mathcal{F}\{g\} \} $$

Donde:
- $\mathcal{F}$ es la **Quantum Fourier Transform (QFT)**.
- $\mathcal{F}^{-1}$ es la **Inverse Quantum Fourier Transform (IQFT)**.
- $\cdot$ es la multiplicación elemento a elemento (interferencia de onda).

## Arquitectura
1.  **Input Encoding:** Tensor $x$ -> Estado Cuántico $|\psi\rangle$.
2.  **QFT:** $|\psi\rangle \xrightarrow{QFT} |\tilde{\psi}\rangle$ (Dominio de Frecuencia).
3.  **Holographic Mask:** $|\tilde{\psi}\rangle \cdot W_{freq}$ (Modulación de Fase/Amplitud).
4.  **IQFT:** $|\tilde{\psi}'\rangle \xrightarrow{IQFT} |\psi'\rangle$ (Dominio Espacial).
5.  **Measurement:** $|\psi'\rangle$ -> Tensor de Salida $y$.

## Ventajas Potenciales
- **Complejidad Computacional:** La convolución se vuelve una multiplicación $O(1)$ (paralela) en el dominio de la frecuencia. La QFT es exponencialmente más rápida que la FFT clásica en hardware cuántico ($O(poly(\log N))$ vs $O(N \log N)$).
- **Interpretabilidad Física:** Los pesos representan directamente filtros de frecuencia (pasa-bajos, pasa-altos, detectores de bordes) de una manera físicamente intuitiva.
- **No-Localidad:** A diferencia de la convolución espacial con kernel pequeño (local), la operación en frecuencia es inherentemente global, permitiendo capturar correlaciones de largo alcance instantáneamente.

## Implementación Actual
- **Clase:** `HolographicConv2d(nn.Module)`
- **Backend:** Híbrido PyTorch + Qiskit (IonQ/Aer).
- **Pesos:** `self.weights_freq` (Parámetro complejo aprendible).

## Referencias
- [[EXP-006_HOLOGRAPHIC_LAYER]]
- [[UNET]]
- [[HARMONIC_ENGINE]]
