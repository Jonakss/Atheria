# APR (Application Performance Repository) para Modelos

El **APR (Application Performance Repository)** es el sistema centralizado de Aetheria para registrar, analizar y comparar el rendimiento de los diferentes modelos de "Ley M". Su objetivo es proporcionar una base de datos viva que guíe la selección de arquitecturas y la optimización de hiperparámetros.

## 1. Métricas Clave

El APR rastrea las siguientes métricas para cada experimento de entrenamiento:

### Rendimiento Computacional
- **Throughput (step/s)**: Velocidad de simulación en pasos por segundo.
- **VRAM Usage (MB)**: Memoria de GPU consumida durante el entrenamiento/inferencia.
- **Training Time**: Tiempo total para alcanzar la convergencia o un número fijo de épocas.
- **Inference Latency (ms)**: Tiempo promedio para calcular un paso de evolución `t -> t+1`.

### Calidad de la Simulación
- **Loss Convergence**: Valor final de la función de pérdida.
- **Energy Conservation Error**: Desviación de la conservación de energía (para modelos unitarios).
- **Complexity Score**: Métrica (aún experimental) para cuantificar la complejidad emergente visual o estadística.
- **Stability**: Capacidad del modelo para mantener estructuras coherentes a largo plazo sin explotar ni decaer al vacío.

## 2. Benchmark de Modelos Actuales

A continuación se presenta una tabla comparativa preliminar de los modelos disponibles (valores estimados en NVIDIA RTX 3090/4090):

| Modelo | Arquitectura | Throughput (128x128) | VRAM (Batch 32) | Estabilidad | Conservación Energía | Uso Recomendado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **U-Net** | CNN Standard | ~450 step/s | ~2.5 GB | Media | No garantizada | Prototipado rápido |
| **U-Net Unitaria** | Unitary CNN | ~380 step/s | ~3.2 GB | Alta | Garantizada (Matemática) | **Producción / Búsqueda de Vida** |
| **MLP** | Fully Connected | >1000 step/s | <1 GB | Baja | No garantizada | Debugging / Baseline |
| **Deep QCA** | Deep CNN | ~200 step/s | ~5.0 GB | Media | No garantizada | Investigación |

## 3. Protocolo de Registro (Logging)

Para mantener el APR actualizado, cada experimento debe registrar sus resultados. Actualmente utilizamos:

1.  **WandB / TensorBoard**: Para métricas en tiempo real (curvas de loss, uso de GPU).
2.  **`AI_DEV_LOG.md`**: Para resúmenes cualitativos y descubrimientos importantes.
3.  **Archivos de Checkpoint**: Los modelos entrenados se guardan con metadatos que incluyen su configuración y métricas finales.

## 4. Análisis de "Edge of Chaos"

El objetivo final del APR no es solo la velocidad, sino encontrar modelos que operen en el "Borde del Caos".
- **Región Ordenada**: Modelos que convergen rápidamente a patrones estáticos o repetitivos simples.
- **Región Caótica**: Modelos que generan ruido blanco o turbulencia sin estructura.
- **Borde del Caos (Target)**: Modelos que muestran estructuras persistentes, móviles (gliders) y reacciones complejas.

> [!TIP]
> Al analizar un nuevo modelo, priorice la **Estabilidad** y la **Complejidad Visual** sobre el `loss` puro. Un loss muy bajo a veces indica que el modelo aprendió la "solución trivial" (vacío).

## 5. Futuras Implementaciones del APR

- [ ] **Dashboard Automático**: Integrar un visualizador de métricas APR en el frontend de Aetheria.
- [ ] **Regression Testing**: Pipeline automático para detectar degradación de rendimiento en nuevos commits.
- [ ] **Model Zoo**: Repositorio de pesos pre-entrenados clasificados por su comportamiento (ej: "Generador de Gliders", "Sopa Primordial", etc.).
