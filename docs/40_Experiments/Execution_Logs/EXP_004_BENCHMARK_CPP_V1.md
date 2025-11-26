Experimento 004: Benchmark Inicial Python vs C++ (Sparse Engine)

Fecha: 2025-11-19
Objetivo: Medir el impacto de mover el almacenamiento de datos a C++ (SparseMap) manteniendo la lógica en Python.

📊 Resultados (Tiempos en Segundos)

Escenario

Python (Control)

C++ V2 (Native Tensors)

Speedup

Micro (100 part, 10 pasos)

0.338s

2.778s

0.12x (Lento)

Medio (500 part, 10 pasos)

1.329s

14.29s

0.09x (Lento)

Grande (1000 part, 5 pasos)

2.820s

14.45s

0.20x (Lento)

🧠 Análisis de Ingeniería

Actualmente, el motor C++ es entre 5x y 10x más lento que Python.

Causa Raíz: "El Problema del Ping-Pong".
Aunque los datos viven en C++, la lógica del bucle step() todavía está orquestada por Python.

Python pide datos a C++.

C++ convierte tensores a Python.

Python calcula.

Python devuelve resultados a C++.

Este viaje de ida y vuelta (Marshaling) en cada paso para cada partícula destruye cualquier ganancia de velocidad.

⚡ Siguiente Paso (Acción Correctiva)

Para lograr el speedup esperado (>100x), debemos eliminar a Python del bucle crítico.

Acción: Migrar la lógica step() y la inferencia (LibTorch) totalmente dentro de C++. Python solo debe decir "Arranca" y "Dame el estado para dibujar".