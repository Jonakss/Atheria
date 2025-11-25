# 🧊 Roadmap Fase 4: Universo Volumétrico (3D Core)

**Objetivo:** Evolucionar la simulación de una superficie 2D a un volumen 3D completo ("El Tanque"), implementando tensores 5D y convoluciones volumétricas.

---

## 1. Fundamentos Conceptuales

**Referencia:** [[20_Concepts/3D_STATE_SPACE_CONCEPT|Conceptualización del Espacio de Estados en 3D]]

La transición a 3D no es meramente visual, sino una expansión fundamental del espacio de fases de la simulación.
- **2D:** Superficie $N \times N$ con estado `d_state`.
- **3D:** Volumen $D \times H \times W$ con estado `d_state`.
- **3D:** Volumen con $(X, Y, Z)$ dimensiones y estado `d_state`.
## 2. Implementación del Motor

### A. Migración de Tensores (PyTorch)
Cambiar la estructura de datos base de 4D a 5D.

- **Actual (4D):** `[Batch, Channels, Height, Width]`
- **Nuevo (5D):** `[Batch, Channels, Depth, Height, Width]`

### B. Adaptación de Redes Neuronales
Migrar la arquitectura U-Net/SNN para operar en 3D.

- Reemplazar `nn.Conv2d` por `nn.Conv3d`.
- Reemplazar `nn.MaxPool2d` por `nn.MaxPool3d`.
- Ajustar capas de normalización (`GroupNorm` soporta 3D, pero requiere verificación de dimensiones).
- Recalcular campos receptivos.

### C. Motor Nativo C++ (Sparse Octree)
El motor nativo (Fase 2) ya contempla coordenadas 3D, pero necesita optimización para vecindades volumétricas.

- **Octree:** Optimizar búsqueda de vecinos en eje Z (arriba/abajo).
- **Hashing:** Verificar colisiones en hash map 3D con mayor densidad.

## 3. Visualización Volumétrica

### A. Proyección Holográfica (AdS/CFT)
Implementar sistemas para visualizar el "Bulk" 3D en pantallas 2D.

- **Slicing:** Ver cortes transversales del cubo (Plano XY a diferentes Z).
- **Raymarching:** Renderizado volumétrico básico (densidad acumulada).
- **Proyecciones:** Integrar valores a lo largo de un eje (ej. suma de energía en Z).

### B. Interfaz de Usuario
- Control de profundidad (Slider Z).
- Rotación de cámara orbital.
- Selección de volumen de interés (VOI) en lugar de ROI.

## 4. Desafíos Computacionales

### A. Explosión de Memoria
Un cubo $128^3$ contiene 2 millones de celdas, comparado con 16k de un plano $128^2$.
- **Solución:** Uso agresivo de Sparse Tensors y cuantización.
- **Chunking:** Simular solo regiones activas del volumen.

### B. Tiempo de Inferencia
Las convoluciones 3D son significativamente más costosas.
- **Solución:** Optimización CUDA y kernels personalizados.

---

**Estado:** Planificación Futura
**Prerrequisitos:**
- [[ROADMAP_PHASE_2|Fase 2: Motor Nativo]] (Infraestructura C++ 3D)
- [[ROADMAP_PHASE_3|Fase 3: Visualización]] (Sistema de renderizado flexible)

---

[[ROADMAP_PHASE_3|← Fase 3]] | **Fase 4 (Futuro)**
