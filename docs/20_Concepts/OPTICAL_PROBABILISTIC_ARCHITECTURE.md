# Arquitectura Óptica Probabilística

## Visión General
Esta arquitectura transforma Aetheria de una simulación digital discreta a una **Simulación Analógica de Onda Continua**. Combina inferencia probabilística (Redes Bayesianas) con principios de computación fotónica (Luz) para modelar la incertidumbre cuántica y la interferencia de manera nativa.

## 1. Computación con Luz (Photonics)
### Concepto Físico
Los chips fotónicos utilizan guías de onda y la interferencia de la luz para realizar multiplicaciones de matrices a la velocidad de la luz.
* **Entrada:** Amplitud y Fase de la luz.
* **Operación:** Interferencia en interferómetros Mach-Zehnder (MZI).
* **Conexión con Aetheria:** El "Motor Polar" (`r, theta`) es isomórfico a la computación óptica.

### Implementación Simulada (`OpticalConv2d`)
Para simular este comportamiento en hardware clásico (GPU), implementamos una capa personalizada:
* **Ruido de Disparo (Shot Noise):** Simula la naturaleza cuántica de los fotones ($N \propto \sqrt{I}$).
* **Precisión Limitada:** Simula la cuantización de DACs/ADCs (ej. 8-bit).
* **Fase/Amplitud:** Opera en el dominio complejo.

```python
class OpticalConv2d(nn.Module):
    def forward(self, x):
        # 1. Cuantización Analógica (DAC)
        x_analog = self.analog_quantization(x)
        # 2. Convolución Física
        out = self.conv(x_analog)
        # 3. Inyección de Ruido Físico
        noise = torch.randn_like(out) * self.noise_level * out.abs().sqrt()
        return out + noise
```

## 2. Inferencia Probabilística (Bayesian U-Net)
### El Universo Borroso
En mecánica cuántica, el estado no es un valor fijo, sino una distribución de probabilidad.
* **Pesos Estocásticos:** Los pesos de la red no son escalares fijos, sino distribuciones Gaussianas $(\mu, \sigma)$.
* **Muestreo:** En cada paso de simulación, se muestrea una "realidad" ligeramente distinta.
* **Visualización:** Las estructuras estables permanecen nítidas; las áreas caóticas se vuelven borrosas o vibrantes, representando visualmente la entropía.

## 3. Formatos Numéricos Exóticos
### Posits vs. BF16
Para maximizar la fidelidad física en hardware limitado:
* **Posits (Unum Type III):** Distribuyen la precisión de forma inteligente, ofreciendo mayor resolución cerca de 0 y 1 (donde vive nuestra probabilidad) y menos en los extremos. Ideal para la unitariedad.
* **BF16 (Brain Floating Point):** Estándar en TPUs, pero con riesgo de deriva numérica en simulaciones físicas largas.

## Roadmap de Implementación
1.  **Fase 1 (Simulación):** Implementar `OpticalConv2d` en PyTorch.
2.  **Fase 2 (Validación):** Entrenar modelos robustos al ruido fotónico.
3.  **Fase 3 (Hardware):** Desplegar en procesadores fotónicos reales (ej. Lightmatter, Quix) cuando estén disponibles.
