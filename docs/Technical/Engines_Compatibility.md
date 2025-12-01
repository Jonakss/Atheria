# Compatibilidad de Motores (Engine Compatibility)

Esta tabla define las capacidades y el estado de implementación de los diferentes motores de física disponibles en Aetheria.

| Motor (Engine) | Representación | Ejecución Nativa (CPU/GPU) | Ejecución Cuántica (QPU) | Estado de Implementación |
| :--- | :--- | :--- | :--- | :--- |
| **Standard (Cartesiano)** | Tensores Complejos `[Re, Im]` | ✅ Sí (PyTorch) | ❌ No | 🟢 Producción (Actual) |
| **Polar (Rotacional)** | Tensores Polares `[Mag, Fase]` | ✅ Sí (PyTorch) | ⚠️ Simulado (Ready) | 🟡 En Desarrollo |
| **Quantum (Híbrido)** | Qubits / Circuitos | ⚠️ Sí (Simulador PennyLane) | ✅ Sí (IBM/Google) | 🔴 Planificado |
| **3D (Volumétrico)** | Tensores 5D | ✅ Sí (Muy pesado) | ❌ No | ⚪ Futuro |

## Detalles de Implementación

### Standard (Cartesiano)
- **Clase:** `Aetheria_Motor`
- **Archivo:** `src/engines/aetheria_engine.py` (o similar)
- **Descripción:** Motor base que utiliza aritmética compleja cartesiana. Optimizado para GPU con PyTorch.

### Polar (Rotacional)
- **Clase:** `Polar_Motor`
- **Archivo:** `src/qca_engine_polar.py`
- **Descripción:** Utiliza representación polar (magnitud y fase) para simular dinámicas rotacionales más naturales.

### Quantum (Híbrido)
- **Clase:** `Hybrid_Motor`
- **Archivo:** `src/qca_engine_pennylane.py` (o similar)
- **Descripción:** Motor experimental que descarga parte del cómputo a simuladores cuánticos o QPUs reales.
