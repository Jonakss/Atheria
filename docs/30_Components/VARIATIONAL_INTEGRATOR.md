# Variational Integrator

**Type:** Physics Module
**Status:** Stable / Core
**Used By:** [[LAGRANGIAN_ENGINE]]

## Overview
El `VariationalIntegrator` (`src/physics/variational_integrator.py`) es una clase reutilizable diseñada para desacoplar las matemáticas de la integración Lagrangiana de la lógica del motor.

## Función Matemática
Resuelve las **Ecuaciones de Euler-Lagrange** para cualquier modelo diferenciable $\mathcal{L}(q, v)$:

$$
\ddot{q} = \left(\nabla_{\dot{q}}^2 \mathcal{L}\right)^{-1} \left( \nabla_q \mathcal{L} - \left(\nabla_q \nabla_{\dot{q}} \mathcal{L}\right) \dot{q} \right)
$$

## Características
- **Agnóstico del Modelo:** Funciona con cualquier `nn.Module` que tome `(q, v)` y retorne un escalar.
- **Autograd:** Utiliza diferenciación automática de PyTorch para calcular fuerzas.
- **Optimización:** Utiliza una aproximación diagonal para $\nabla_{\dot{q}}^2 \mathcal{L}$ para viabilidad en grids grandes.

## Uso

```python
from src.physics.variational_integrator import VariationalIntegrator

# Instanciar con un modelo
integrator = VariationalIntegrator(model)

# Evolucionar un paso
q_next, v_next = integrator.step(q, v, dt=0.01)
```
