# Lagrangian Engine (LNN)

**Type:** Physics Engine
**Status:** Experimental / Prototype
**Version:** 1.0.0
**Related:** [[VARIATIONAL_INTEGRATOR]], [[HARMONIC_ENGINE]]

## Resumen
El **Lagrangian Engine** introduce un cambio de paradigma en Aetheria. En lugar de simular la física mediante reglas explícitas (Hamiltoniano / Ecuaciones diferenciales hardcodeadas), este motor utiliza una **Lagrangian Neural Network (LNN)** para aprender la función Lagrangiana $\mathcal{L}(q, \dot{q})$ del sistema.

La evolución temporal **emerge** de minimizar la Acción $S = \int \mathcal{L} dt$ utilizando las ecuaciones de Euler-Lagrange, resueltas numéricamente por el [[VARIATIONAL_INTEGRATOR]].

## Componentes

### 1. Lagrangian Network (`src/models/lagrangian_net.py`)
Una red neuronal que aproxima la densidad Lagrangiana:
- **Input:** Estado $q$ y Velocidad $v$ (concatenados).
- **Output:** Escalar $\mathcal{L}$ (Densidad de Acción por celda).
- **Arquitectura:** CNN con kernels 1x1 (localidad) para permitir inversión eficiente del Hessiano.

### 2. Variational Integrator (`src/physics/variational_integrator.py`)
Un módulo agnóstico que resuelve:
$$ \frac{d}{dt} \frac{\partial \mathcal{L}}{\partial \dot{q}} = \frac{\partial \mathcal{L}}{\partial q} $$
Calcula la aceleración $\ddot{q}$ invirtiendo la matriz Hessiana (masa efectiva) y aplica integración simpléctica.

### 3. Lagrangian Engine (`src/engines/lagrangian_engine.py`)
Implementa el `EngineProtocol` estándar de Aetheria.
- **Configuración:** `ENGINE_TYPE: "LAGRANGIAN"`
- **Estado:** Mantiene $q$ (parte Real) y $v$ (parte Imaginaria del campo complejo).

## Uso
Para usar este motor en un experimento, añadir a la configuración:

```json
{
  "ENGINE_TYPE": "LAGRANGIAN",
  "dt": 0.1,
  "MODEL_PARAMS": {
      "d_state": 3
  }
}
```

## Estado Actual
- Implementación base funcional con aproximación diagonal del Hessiano.
- Integrado en `MotorFactory`.
- Tests básicos de conservación de energía (en oscilador no entrenado) exitosos.

## Próximos Pasos
- Entrenar la LNN para que aprenda leyes físicas complejas (conservación de energía real).
- Implementar términos de interacción espacial (convoluciones > 1x1) con manejo eficiente del Hessiano no diagonal.
