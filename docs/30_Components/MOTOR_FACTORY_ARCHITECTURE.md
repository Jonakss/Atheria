# Arquitectura de Fábrica de Motores (Motor Factory)

## Resumen
La **Motor Factory** (`src/motor_factory.py`) es el componente responsable de instanciar el motor de física adecuado para cada experimento en Aetheria. Permite desacoplar la lógica de creación del motor del resto del servidor, facilitando la incorporación de nuevos motores (como Polar, Quantum, Lattice) sin modificar los handlers principales.

## Motivación
Anteriormente, Aetheria usaba por defecto el motor cartesiano (`Aetheria_Motor` en `qca_engine.py`). Con la introducción de nuevas topologías y físicas (Polar, Lattice, Quantum), se hizo necesario un mecanismo centralizado para seleccionar e inicializar el motor correcto basado en la configuración del experimento.

## Componentes

### 1. Motor Factory (`src/motor_factory.py`)
Expone una única función pública:
```python
def get_motor(config, model, device) -> MotorInterface
```
- **Entradas:**
    - `config`: Objeto o diccionario con la configuración del experimento (debe contener `ENGINE_TYPE`).
    - `model`: El modelo neuronal (operador) cargado.
    - `device`: Dispositivo de ejecución (CPU/CUDA).
- **Salida:** Instancia del motor seleccionado.

### 2. Tipos de Motores Soportados (`ENGINE_TYPE`)

| Tipo | Clave Config | Clase | Descripción |
| :--- | :--- | :--- | :--- |
| **Cartesiano** | `CARTESIAN` | `Aetheria_Motor` | Motor estándar. Grid 2D, condiciones de contorno toroidales. |
| **Polar** | `POLAR` | `PolarEngine` | Topología rotacional. Coordenadas polares (r, θ). Estabilidad mejorada. |
| **Quantum** | `QUANTUM` | `Aetheria_Motor`* | Híbrido. Procesamiento en QPU (actualmente simulación clásica con flags). |
| **Lattice** | `LATTICE` | `LatticeEngine` | Gauge Theory (QCD). En desarrollo. |

### 3. Integración en Flujo de Trabajo

1.  **Creación de Experimento (Frontend):**
    - El usuario selecciona el "Motor Físico" en el panel lateral (`LabSider.tsx`).
    - El valor se envía como `ENGINE_TYPE` al backend.
    - Se persiste en `config.json`.

2.  **Carga de Experimento (Backend):**
    - `handle_load_experiment` lee `config.json`.
    - Llama a `get_motor(config, ...)` en lugar de instanciar `Aetheria_Motor` directamente.
    - El motor retornado se guarda en `g_state['motor']`.

3.  **Visualización:**
    - `pipeline_viz.py` y `viz/core.py` detectan el tipo de motor o estado (Duck Typing).
    - Si es Polar, mapea Fase -> Color directamente.
    - Si es Cartesiano, calcula `atan2(Im, Re)`.

## Extensibilidad
Para agregar un nuevo motor:
1.  Implementar la clase del motor (debe tener `state.psi`, `evolve_step`, etc.).
2.  Importar la clase en `src/motor_factory.py`.
3.  Agregar la condición en `get_motor`.
4.  Agregar la opción en el frontend (`LabSider.tsx`).

## Referencias
- [[ENGINE_COMPATIBILITY_MATRIX]]
- [[NATIVE_ENGINE_CORE]]
