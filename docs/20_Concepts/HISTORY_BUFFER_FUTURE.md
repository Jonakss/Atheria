# Historial/Buffer de Simulación - Implementación Futura

## 📋 Resumen

Sistema de historial/buffer para almacenar frames de simulación y permitir navegación temporal (rewind, replay, análisis histórico).

## 🎯 Objetivos

1. **Buffer Circular:** Almacenar últimos N frames en memoria para navegación rápida
2. **Historial Persistente:** Guardar frames en disco para análisis posterior
3. **Navegación Temporal:** Permite retroceder, avanzar, y saltar a cualquier paso guardado
4. **Análisis Comparativo:** Comparar estados en diferentes pasos temporales

## 🔍 Estado Actual

### Componentes Existentes

1. **`src/managers/history_manager.py`**: 
   - Manager para guardar frames en disco
   - Método `add_frame()` para agregar frames al historial
   - Soporte para guardar/cargar archivos de historia

2. **Sistema de Snapshots**:
   - Captura snapshots del estado psi cada N pasos
   - Almacenamiento limitado (últimos 500 snapshots)
   - Usado para análisis t-SNE

3. **Handlers Backend**:
   - `handle_enable_history`: Habilitar/deshabilitar guardado de historia
   - `handle_save_history`: Guardar historial actual a archivo
   - `handle_load_history_file`: Cargar historial desde archivo

### Estado de Implementación

- ✅ **Manager básico:** Implementado (`history_manager.py`)
- ✅ **Handlers backend:** Implementados
- ⚠️ **Frontend:** Parcialmente implementado (HistoryView existe pero necesita integración)
- ❌ **Buffer circular en memoria:** No implementado
- ❌ **Navegación temporal:** No implementado
- ❌ **UI de rewind/replay:** Pendiente

## 🚀 Plan de Implementación Futura

### Fase 1: Buffer Circular en Memoria

```python
class SimulationBuffer:
    """
    Buffer circular para almacenar frames recientes en memoria.
    Permite acceso rápido a los últimos N frames sin I/O de disco.
    """
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
        self.current_index = 0
        
    def add_frame(self, frame_data: dict):
        """Agregar frame al buffer (circular)."""
        pass
        
    def get_frame(self, step: int) -> dict | None:
        """Obtener frame por número de paso."""
        pass
        
    def get_recent_frames(self, count: int) -> list[dict]:
        """Obtener últimos N frames."""
        pass
```

### Fase 2: Navegación Temporal en Frontend

- Control de timeline para saltar a cualquier paso
- Botones de rewind/forward
- Indicador de posición actual en el historial
- Vista previa de frames guardados

### Fase 3: Historial Persistente Mejorado

- Compresión de frames guardados
- Indexación rápida por paso temporal
- Búsqueda por metadata (step, timestamp, etc.)
- Exportar/importar historiales completos

## 📝 Notas de Diseño

### Consideraciones de Memoria

- Buffer circular limitado (ej: últimos 1000 frames)
- Historial persistente con compresión (LZ4, zlib)
- Opción de guardar solo frames clave (cada N pasos)

### Integración con Live Feed

- Cuando live feed está activo: buffer + historial completo
- Cuando live feed está pausado: solo guardar frames cada X pasos
- Historial persistente independiente del buffer en memoria

### Formatos de Almacenamiento

- **Buffer en memoria:** Lista de dicts (rápido, volátil)
- **Historial en disco:** JSON comprimido o formato binario optimizado
- **Metadata:** SQLite para búsquedas rápidas

## 🔗 Referencias

- `src/managers/history_manager.py`: Implementación actual
- `src/pipelines/pipeline_server.py`: Handlers de historial (líneas ~376-382, ~2131-2182)
- `frontend/src/modules/Dashboard/components/HistoryView.tsx`: UI pendiente de integración
- `src/server/server_handlers.py`: Handlers de historial (líneas ~1171, ~1200)

## 📅 Estado

**Fecha de Nota:** 2024
**Estado:** Pendiente de implementación
**Prioridad:** Media

---

*Nota: Este documento se actualizará cuando se implemente el sistema de historial/buffer completo.*

