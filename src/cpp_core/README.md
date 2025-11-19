# Atheria Core - Núcleo C++ de Alto Rendimiento

Este directorio contiene el código fuente C++ para las extensiones de alto rendimiento de Atheria 4.

## Estructura

```
cpp_core/
├── include/          # Headers C++ (.h)
├── src/             # Código fuente C++ (.cpp)
├── tests/           # Tests unitarios C++
└── README.md        # Este archivo
```

## Compilación

El módulo se compila automáticamente al instalar el proyecto:

```bash
pip install -e .
```

O usando setup.py directamente:

```bash
python setup.py build_ext --inplace
```

## Requisitos

- CMake >= 3.15
- Compilador C++ con soporte C++17 (GCC, Clang, o MSVC)
- Python 3.8+
- PyBind11 >= 2.10.0

## Verificación

Después de compilar, puedes verificar que el módulo funciona:

```bash
python scripts/test_cpp_binding.py
```

## Componentes Actuales

### SparseMap

Clase de mapa disperso de alto rendimiento para el motor de simulación.

### Función add()

Función simple de prueba para verificar que los bindings funcionan.

## Próximos Componentes

- Motor de física disperso nativo
- Operaciones vectorizadas optimizadas
- Integración con CUDA (futuro)

