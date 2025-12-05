# 🏗️ Tareas Arquitectónicas Pendientes - Atheria 4

**Creado:** 2025-12-05  
**Estado:** 📋 **Planificación**

---

## 🔴 PRIORIDAD ALTA - Arquitectura de Motores

### 1. Refactorizar Filosofía de Emergencia vs Inyección

**Problema:**  
El código actual de `SparseHarmonicEngine._ingest_dense_state()` **inyecta partículas** manualmente al diccionario `matter` cuando debería permitir que las estructuras **emerjan** de la evolución del campo.

**Decisión de Diseño (Confirmada por Usuario):**
- Todos los engines deberían trabajar con **campo denso** (como CartesianEngine)
- Si se usa modelo híbrido (campo + partículas dispersas), las "partículas" en `matter` deberían **emerger durante la evolución**, no al reset
- Esto es importante para **inferencia en tiempo real**

**Archivos Afectados:**
- `src/engines/harmonic_engine.py` - `SparseHarmonicEngine._ingest_dense_state()`, `reset_to_initial_from_dense()`
- Posiblemente crear variante híbrida que combine campo denso con estructuras emergentes

**Tareas:**
- [ ] Documentar claramente la filosofía de "emergencia" en los docstrings
- [ ] Refactorizar `_ingest_dense_state()` para que sea opcional o eliminarla
- [ ] Asegurar que `SparseHarmonicEngine` pueda funcionar 100% con campo denso
- [ ] Si se mantiene híbrido, las partículas deben emerger en `step()` no en reset

---

### 2. Holographic como Capa de Visualización (No Motor)

**Problema:**  
`HolographicEngine` actualmente es un motor separado que extiende `CartesianEngine`. Sin embargo, conceptualmente es una **capa de visualización** que puede aplicarse a **cualquier motor**.

**Decisión de Diseño:**
- Holographic = Cartesian + proyección al Bulk 3D (AdS/CFT)
- Debería ser una **opción de visualización**, no un tipo de motor
- El `get_bulk_state()` podría moverse a un módulo de visualización

**Archivos Afectados:**
- `src/engines/holographic_engine.py` - Evaluar si debe ser motor o viz layer
- `frontend/src/components/` - Selector de motores vs selector de visualización

**Tareas:**
- [ ] Evaluar si mover `HolographicEngine` a `src/pipelines/viz/holographic.py`
- [ ] Frontend: Separar "Tipo de Motor" de "Modo de Visualización"
- [ ] Actualizar documentación para clarificar la distinción

---

### 3. Frontend: Lista de Motores Disponibles

**Problema:**  
El selector de motores en el frontend necesita actualizarse para reflejar correctamente:
- Motores reales: Cartesian, Polar, Harmonic (y Native como modo)
- Visualizaciones: Standard, Holographic (proyección 3D)

**Tareas:**
- [ ] Actualizar `LabSider.tsx` para mostrar engines correctos
- [ ] Agregar selector separado para "Visualización Holográfica"
- [ ] Sync con backend para validar engines disponibles

---

## 🟡 PRIORIDAD MEDIA - Análisis y Visualización

### 4. Agregar Análisis UMAP y t-SNE

**Contexto:**  
Análisis de reducción de dimensionalidad para visualizar el espacio de estados del campo cuántico.

**UMAP (Uniform Manifold Approximation and Projection):**
- Preserva estructura local y global
- Útil para ver clusters de estados similares

**t-SNE (t-distributed Stochastic Neighbor Embedding):**
- Excelente para visualizar clusters
- Más costoso computacionalmente

**Tareas:**
- [ ] Agregar dependencias: `umap-learn`, `scikit-learn`
- [ ] Crear módulo `src/pipelines/analysis/dimensionality.py`
- [ ] Integrar en frontend como panel de análisis
- [ ] Permitir samplear estados durante simulación para análisis

---

## 📚 Referencias

- [[ATHERIA_4_MASTER_BRIEF]] - Filosofía del proyecto
- [[TECHNICAL_ARCHITECTURE_V4]] - Arquitectura actual
- [[AI_DEV_LOG]] - Historial de decisiones

---

**Notas:**
> Las estructuras deben EMERGER de la evolución del campo, no ser inyectadas.  
> El Principio Holográfico es una forma de VER la información, no de calcularla.
