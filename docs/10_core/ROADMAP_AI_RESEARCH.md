# 🧠 Roadmap Investigación IA: The Brain (Ley M)

**Objetivo:** Desarrollar y evolucionar "Ley M", la red neuronal que actúa como las leyes fundamentales de la física en Atheria, buscando arquitecturas que favorezcan la emergencia de complejidad.

---

## 1. Arquitecturas de Modelos

**Referencia:** [[MASSIVE_INFERENCE_ARCHITECTURE|Arquitectura de Inferencia]]

### A. Redes Neuronales de Pulsos (SNN)
Explorar redes que operan con eventos discretos (spikes) para mayor eficiencia y realismo biológico/físico.
- **Spiking U-Net:** Adaptar la arquitectura U-Net para usar neuronas LIF (Leaky Integrate-and-Fire).
- **Eficiencia Energética:** Aprovechar la escasez (sparsity) de los spikes.

### B. Transformers & Attention
- **Vision Transformers (ViT):** Aplicar mecanismos de atención para capturar dependencias de largo alcance en el grid.
- **Physics-Informed Attention:** Restringir la atención a conos de luz causales.

### C. Variantes de U-Net
- **Unitary U-Net:** Garantizar la preservación de norma (energía) mediante matrices ortogonales.
- **3D U-Net:** (Para Fase 4/5) Adaptar convoluciones para volúmenes.

---

## 2. Curriculum Learning (Evolución)

**Referencia:** [[PROGRESSIVE_LEARNING|Aprendizaje Progresivo]]

### A. Definición de Épocas
Formalizar las etapas de entrenamiento para guiar la complejidad.
1.  **Vacío:** Aprender a mantener el vacío estable (eliminar ruido).
2.  **Partículas:** Aprender a formar excitaciones estables (solitones).
3.  **Interacción:** Aprender reglas de colisión y dispersión.
4.  **Estructura:** Formación de agregados complejos.

### B. Epoch Detector
- **Métricas de Complejidad:** Desarrollar métricas robustas para detectar cambios de fase (ej. Dimensión Fractal, Entropía de Shannon).
- **Trigger Automático:** El sistema debe cambiar los hiperparámetros (ruido, learning rate) automáticamente al detectar estancamiento o hitos.

---

## 3. Funciones de Pérdida (The Laws)

### A. Physics Loss
Incorporar restricciones físicas directamente en la función de pérdida.
- **Hamiltonian Loss:** Penalizar violaciones de conservación de energía.
- **Symmetry Loss:** Penalizar violaciones de simetrías (rotación, traslación, CPT).

### B. Information Loss
- **Variational Information Bottleneck:** Forzar al modelo a comprimir información, quedándose solo con lo relevante (causalidad).

---

## 4. Meta-Learning & Auto-ML

- **Neural Architecture Search (NAS):** Dejar que la IA evolucione su propia arquitectura.
- **Hyperparameter Optimization:** Búsqueda automática de los mejores parámetros de entrenamiento.

---

**Estado:** En Progreso (Investigación Continua)
**Relación:** Transversal a todas las fases (el "Cerebro" del sistema).
