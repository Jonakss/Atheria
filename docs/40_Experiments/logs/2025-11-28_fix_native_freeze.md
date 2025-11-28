# 📝 Log: Corrección de Congelamiento en Motor Nativo

**Fecha:** 2025-11-28
**Autor:** Antigravity Agent
**Estado:** ✅ Corregido

## 🚨 Problema
El usuario reportó que el motor nativo "se tranca" (se congela).
-   **Síntoma:** La simulación deja de responder o se vuelve extremadamente lenta.
-   **Causa Raíz:** La conversión de estado disperso (C++) a denso (Python) en `native_engine_wrapper.py` se realizaba iterando sobre las partículas en Python. Con muchas partículas, este bucle bloqueaba el GIL, impidiendo que el bucle de eventos procesara mensajes WebSocket (heartbeats), causando desconexión o freeze aparente.

## 🛠️ Solución Implementada
Se optimizó `src/pipelines/core/simulation_loop.py` para usar el **Fast Path** de visualización nativa.

1.  **Bypass de Conversión Lenta:**
    -   Se detecta si el motor es nativo y soporta `get_visualization_data`.
    -   Se llama directamente a `motor.get_visualization_data(viz_type)`, que invoca `compute_visualization` en C++.
    -   Esto retorna un tensor denso [H, W] calculado eficientemente en C++ (OpenMP), evitando el bucle lento de Python.

2.  **Lógica de Fallback:**
    -   Si el tipo de visualización no es soportado por C++ (ej: "entropy", "flow"), se usa el camino lento (conversión a denso + viz en Python).
    -   Se mantiene la generación de estado denso (`psi`) solo cuando es estrictamente necesario (ej: `EpochDetector` cada 50 pasos).

3.  **Corrección de Flujo:**
    -   Se ajustó la lógica para permitir enviar frames incluso si `psi` es `None` (cuando se usa el Fast Path).

## ⚠️ Limitaciones
-   Las visualizaciones avanzadas ("entropy", "flow") seguirán siendo lentas en el motor nativo hasta que se implementen en C++.
-   El historial en motor nativo será "solo visual" (sin estado cuántico `psi` guardado) para la mayoría de los frames, lo cual es aceptable dado el fix anterior de rewind.

## 🔗 Referencias
-   `src/pipelines/core/simulation_loop.py`
-   `src/engines/native_engine_wrapper.py`
-   `src/cpp_core/src/bindings.cpp`
