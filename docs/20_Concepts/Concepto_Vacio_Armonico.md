id: concepto_vacio_armonico
tipo: concepto_fisico
tags: [qft, optimizacion, motor, infinitive_universe]

🌊 Vacío Armónico (Harmonic Vacuum)

Definición

El Vacío Armónico es una técnica de generación procedural utilizada en [[Atheria 4]] para simular el estado base del universo sin consumir memoria RAM. Reemplaza al "vacío nulo" (ceros) y al "ruido blanco" (random).

Fundamento Físico (QED)

En la Teoría Cuántica de Campos, el vacío no está vacío; está lleno de campos oscilando en su estado de mínima energía. Estas fluctuaciones son necesarias para:

Permitir el movimiento de partículas (romper la simetría de traslación).

Proveer un "baño térmico" con el cual interactuar.

Implementación Matemática

Se calcula como la superposición de $N$ ondas planas estacionarias para cada canal $d$:

$$\Psi(x,y,z,t) = \sum_{i=1}^{N} A_i \cdot \sin(\vec{k}_i \cdot \vec{r} - \omega_i t + \phi_i)$$

$\vec{k}$: Vector de onda (frecuencia espacial).

$\omega$: Frecuencia temporal.

$\phi$: Fase aleatoria determinista.

Ventajas Técnicas

Determinismo: get_state(x, y, t) siempre devuelve el mismo valor, permitiendo reproducibilidad.

Infinidad: Se puede calcular para cualquier coordenada $(x, y, z)$ sin límites.

Cero Memoria: No se guarda en arrays; se computa "on-the-fly" (al vuelo).

Relación con Otros Sistemas

Es utilizado por el [[SparseQuantumEngine]] para rellenar los huecos entre la materia.

Interactúa con la [[Ley M]] durante la inferencia.