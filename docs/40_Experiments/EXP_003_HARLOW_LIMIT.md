# Experimento 03: El Límite de Harlow (Complejidad desde la Simplicidad)

## 🎯 Objetivo Científico
Validar la hipótesis de que un sistema cuántico fundamentalmente simple (con un espacio de estados casi trivial) puede generar una complejidad visual y dinámica indistinguible del caos para un "observador interno".

Este experimento se basa en la teoría reciente de gravedad cuántica (Harlow et al., MIT) que sugiere que si el universo es cerrado, su espacio de Hilbert podría ser unidimensional (un solo estado estático), y la complejidad que percibimos es producto del *coarse-graining* (baja resolución) del observador. Ver [[The_Harlow_Limit_Theory]] para más detalles teóricos.

## 🧪 Hipótesis en Aetheria
Si configuramos nuestro Motor QCA para que sea **perfectamente unitario** y **cerrado** (sin decaimiento, sin ruido externo), el estado matemático global del sistema debería permanecer constante o cíclico. Sin embargo, si observamos solo una proyección (visualización 2D de fase/densidad), deberíamos ver patrones complejos emerger.

**La paradoja a demostrar:**
> $\frac{d}{dt} |\Psi_{global}|^2 \approx 0$  (El universo es estático matemáticamente)
> $Complejidad(\text{Visual}) \gg 0$ (El universo parece vivo para nosotros)

## ⚙️ Configuración del Experimento

### 1. Parámetros Físicos (`src/config.py`)
* **Modelo:** `UNetUnitary` (Estrictamente conservativo).
* **Decaimiento (`GAMMA_DECAY`):** `0.0` (Sistema cerrado, sin pérdida de energía).
* **Ruido Inicial:** Mínimo posible, solo para romper la simetría perfecta inicial.

### 2. Nuevas Métricas a Implementar
Para medir esta paradoja, necesitamos instrumentación específica en `src/trainer.py` o `src/pipeline_viz.py`:

* **Fidelidad Global ($F$):** Mide cuánto cambia el estado cuántico total respecto al inicio.
    $$F(t) = |\langle \Psi(0) | \Psi(t) \rangle|^2$$
    * *Expectativa:* $F(t)$ debe mantenerse muy alto (cercano a 1).

* **Entropía de Enlazamiento (Subsistema):** Dividimos el grid en dos mitades A y B. Calculamos la entropía de Von Neumann de la mitad A.
    $$S_A = -Tr(\rho_A \log \rho_A)$$
    * *Expectativa:* $S_A$ debe crecer, indicando que aunque el todo es simple, las partes se vuelven complejas y entrelazadas.

## 📝 Plan de Ejecución

1.  **Entrenamiento:** Entrenar una "Ley M" con una función de pérdida que *penalice* el cambio en la energía total pero *premie* la entropía local.
2.  **Simulación:** Ejecutar el modelo entrenado por 10,000 pasos.
1.  **Entrenamiento:** Entrenar una "Ley M" (nuestro modelo de dinámica del universo, ver [[referencia a Ley M]]) con una función de pérdida que *penalice* el cambio en la energía total pero *premie* la entropía local.
    * Graficar la Fidelidad vs. Tiempo.
    * Graficar la Entropía Visual vs. Tiempo.
    * Si las gráficas divergen (Fidelidad alta, Entropía alta), habremos replicado el "Efecto Harlow".

## 🔗 Conexión con AdS/CFT
Si este experimento tiene éxito, refuerza la interpretación de Aetheria como un modelo de juguete holográfico. La "simpleza" del estado global corresponde al interior del agujero negro (o universo cerrado), y la "complejidad" visual corresponde a la proyección holográfica en la frontera (nuestra pantalla).
