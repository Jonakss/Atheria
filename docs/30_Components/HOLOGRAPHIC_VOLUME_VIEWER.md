# Holographic Visualization - Complete Guide

## Two Different Concepts

### 1. HolographicEngine Bulk (Physics Engine)
**Qué es**: Motor físico que implementa el principio holográfico  
**Dónde**: `HolographicEngine` (hereda de `CartesianEngine`)  
**Método**: `motor.get_bulk_state()` - Proyección física del motor  
**Uso**: Solo para experimentos creados con `engine_type='HOLOGRAPHIC'`

```python
# Backend
motor = HolographicEngine(...)
bulk = motor.get_bulk_state()  # [1, D, H, W] - bulk físico
```

```typescript
// Frontend - Solo HolographicEngine
sendMessage({ type: 'get_bulk_volume' });
// Responde: { type: 'bulk_volume_data', payload: {...} }
```

---

### 2. Generic Holographic Projection (Visualization Technique)
**Qué es**: Técnica de visualización para CUALQUIER motor 2D  
**Dónde**: `src/pipelines/viz/holographic_projection.py`  
**Método**: `project_2d_to_3d_holographic()` - Scale-Space projection  
**Uso**: Funciona con Cartesian, Polar, Harmonic, Lattice, etc.

**El concepto (del paper)**: Un campo cuántico 2D contiene información completa. Usando Scale-Space (Gaussian blur progresivo), podemos generar capas de "profundidad" que representan diferentes escalas de renormalización.

```python
# Backend - Funciona con CUALQUIER motor
from ...pipelines.viz.holographic_projection import visualize_as_hologram

volume = visualize_as_hologram(motor, depth=8, use_phase=False)
# [1, D, H, W] - proyección holográfica del estado 2D
```

```typescript
// Frontend - Cualquier motor
sendMessage({ 
  type: 'get_holographic_projection',
  depth: 8,
  use_phase: false  // true para proyectar fase en vez de magnitud
});
// Responde: { type: 'holographic_projection_data', payload: {...} }
```

---

## Cómo Funciona la Proyección Genérica

### Scale-Space Technique

1. **Capa Z=0 (Boundary)**: Estado original sin procesar
   - UV (ultravioleta): Alta frecuencia, detalles finos
   - Representa el "boundary" del AdS/CFT

2. **Capas Z=1,2,...,N (Bulk)**: Gaussian blur progresivo
   - Sigma aumenta con Z: `sigma = z * 0.5 + 0.5`
   - IR (infrarrojo): Baja frecuencia, estructuras gruesas
   - Representa el "bulk" a diferentes escalas

### Interpretación Física

- **Profundidad Z** ≈ Escala de renormalización
- **Z pequeño** → Alta energía, detalles cuánticos
- **Z grande** → Baja energía, fenómenos emergentes clásicos

---

## Uso en Frontend

### Opción 1: Mostrar Automático por Tipo de Motor

```tsx
// HolographicEngine -> usar get_bulk_volume (física del motor)
if (engineType === 'HOLOGRAPHIC') {
  sendMessage({ type: 'get_bulk_volume' });
}

// Otros motores 2D -> usar get_holographic_projection (visualización)
else if (['CARTESIAN', 'POLAR', 'HARMONIC', 'LATTICE'].includes(engineType)) {
  sendMessage({ type: 'get_holographic_projection', depth: 8 });
}
```

### Opción 2: Toggle Manual

```tsx
<button onClick={() => {
  if (engineType === 'HOLOGRAPHIC') {
    sendMessage({ type: 'get_bulk_volume' });
  } else {
    sendMessage({ type: 'get_holographic_projection', depth: 8 });
  }
}}>
  Ver Holograma 3D
</button>
```

### Manejando la Respuesta

```tsx
useEffect(() => {
  if (lastMessage?.type === 'bulk_volume_data' || 
      lastMessage?.type === 'holographic_projection_data') {
    setVolumeData({
      data: lastMessage.payload.volume_data,
      depth: lastMessage.payload.depth,
      height: lastMessage.payload.height,
      width: lastMessage.payload.width
    });
  }
}, [lastMessage]);
```

---

## Configuración Avanzada

### Parámetros de Proyección

```python
project_2d_to_3d_holographic(
    state_2d=psi,
    depth=8,          # Número de capas (default: 8)
    sigma_scale=0.5,  # Factor de suavizado (0.5=suave, 2.0=agresivo)
    device='cpu'
)
```

### Proyectar Fase vs Magnitud

```typescript
// Magnitud (default) - estructura espacial
sendMessage({ type: 'get_holographic_projection', use_phase: false });

// Fase - información de coherencia cuántica
sendMessage({ type: 'get_holographic_projection', use_phase: true });
```

---

## Resumen

| Aspecto | HolographicEngine Bulk | Generic Projection |
|---------|------------------------|-------------------|
| **¿Para qué motores?** | Solo `HOLOGRAPHIC` | Todos los 2D |
| **¿Qué proyecta?** | Bulk físico del motor | Visualización Scale-Space |
| **Handler** | `get_bulk_volume` | `get_holographic_projection` |
| **Mensaje response** | `bulk_volume_data` | `holographic_projection_data` |
| **Parámetros** | Ninguno | `depth`, `use_phase` |

**Recomendación**: Usar el viewer genérico (`get_holographic_projection`) para todos los motores 2D como visualización adicional. Reservar `get_bulk_volume` para el `HolographicEngine` cuando quieras el bulk físico verdadero.
