# 🧠 Gestión de Memoria - Atheria 4

**Fecha:** 2024-12-XX  
**Objetivo:** Documentar y optimizar el manejo de memoria en todo el proyecto.

---

## 📊 Análisis de Memoria

### Áreas Críticas Identificadas

#### 1. **Trainers - `psi_history` sin límite**
- **Problema:** En `QC_Trainer_v4.train_episode()`, `psi_history` crece linealmente con `qca_steps` (puede ser 100+ pasos).
- **Impacto:** Alto - Cada elemento es un tensor completo `[1, H, W, d_state]`.
- **Solución:** Solo guardar estados necesarios o usar subsampling.

#### 2. **Snapshots sin liberación explícita**
- **Problema:** Los snapshots se limitan a 500, pero se clonan tensores completos que pueden quedarse en GPU.
- **Impacto:** Medio - Depende del tamaño del grid y d_state.
- **Solución:** Mover a CPU explícitamente y limpiar cuando se excede el límite.

#### 3. **Frontend - Estado acumulativo**
- **Problema:** `simData` y `allLogs` pueden crecer indefinidamente.
- **Impacto:** Medio - Depende del tiempo de uso.
- **Solución:** Limitar tamaño máximo y rotar logs.

#### 4. **Motor Global sin limpieza**
- **Problema:** `g_state['motor']` no se libera explícitamente al cambiar de modelo.
- **Impacto:** Alto - Puede mantener referencias a tensores grandes.
- **Solución:** Limpiar motor anterior antes de cargar uno nuevo.

#### 5. **ConvLSTM Memory States**
- **Problema:** `h_state` y `c_state` pueden crecer si no se resetean correctamente.
- **Impacto:** Medio - Solo para modelos ConvLSTM.
- **Solución:** Resetear estados de memoria cuando sea necesario.

#### 6. **Simulation History**
- **Problema:** Límite de 500 frames, pero cada frame puede tener `map_data` grande.
- **Impacto:** Medio - Ya tiene downsampling, pero puede mejorarse.
- **Solución:** ✅ Ya implementado correctamente con downsampling.

---

## 🔧 Optimizaciones Implementadas

### Backend - Python

1. **Limpieza explícita de motor al cargar nuevo modelo**
2. **Mover snapshots a CPU explícitamente**
3. **Liberar `psi_history` después de calcular pérdida**
4. **Garbage collection periódico para tensores huérfanos**

### Frontend - React/TypeScript

1. **Límite máximo para logs acumulativos**
2. **Rotación de logs más antiguos**
3. **Límite máximo para `simData` si se acumula**

### C++ Engine

1. **Verificar que destructores liberan recursos correctamente**
2. **Usar smart pointers para gestión automática**

---

## 📝 Notas de Implementación

- Los snapshots deben moverse a CPU antes de almacenarse.
- Los tensores intermedios deben usar `torch.no_grad()` cuando sea posible.
- Los modelos ConvLSTM deben resetear estados de memoria periódicamente.
- El frontend debe limitar el tamaño de datos acumulativos.

---

**Estado:** ⏳ Revisión y optimizaciones en progreso

