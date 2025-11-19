# 📋 Responsabilidades de Cada Carpeta en la Documentación

> Guía clara sobre qué tipo de documentación pertenece a cada carpeta

---

## 📁 10_core/ - Documentación Core del Proyecto

**Responsabilidad**: Documentación fundamental, arquitectura general, y filosofía del proyecto.

### Qué incluir:
- ✅ Brief maestros y visión del proyecto
- ✅ Glosarios y terminología
- ✅ Arquitectura técnica general
- ✅ Roadmaps y planificación
- ✅ Guías de aprendizaje estructuradas
- ✅ Diseño de sistemas distribuidos

### Qué NO incluir:
- ❌ Detalles de implementación específicos (van a `30_Components/`)
- ❌ Resultados de experimentos (van a `40_Experiments/`)
- ❌ Conceptos físicos puros (van a `20_Concepts/`)

### Ejemplo:
- `ATHERIA_4_MASTER_BRIEF.md` - Visión general ✅
- `TECHNICAL_ARCHITECTURE_V4.md` - Arquitectura general ✅
- `ROADMAP_PHASE_1.md` - Planificación ✅

---

## 📁 20_Concepts/ - Conceptos Teóricos

**Responsabilidad**: Conceptos físicos, matemáticos y teóricos fundamentales.

### Qué incluir:
- ✅ Conceptos físicos (vacío cuántico, QCA, etc.)
- ✅ Fundamentos matemáticos
- ✅ Teorías y principios base
- ✅ Definiciones teóricas puras

### Qué NO incluir:
- ❌ Implementación técnica (van a `30_Components/`)
- ❌ Resultados experimentales (van a `40_Experiments/`)
- ❌ Guías prácticas (van a `10_core/` o `40_Experiments/`)

### Ejemplo:
- `HARMONIC_VACUUM_CONCEPT.md` - Concepto físico ✅

---

## 📁 30_Components/ - Componentes Técnicos

**Responsabilidad**: Documentación técnica de componentes, sistemas y módulos implementados.

### Qué incluir:
- ✅ Arquitectura de componentes específicos
- ✅ APIs y interfaces
- ✅ Optimizaciones técnicas
- ✅ Sistemas y módulos
- ✅ Guías de uso de componentes
- ✅ Análisis técnicos (precisión, rendimiento de componentes)

### Qué NO incluir:
- ❌ Resultados de experimentos/benchmarks (van a `40_Experiments/`)
- ❌ Filosofía general del proyecto (van a `10_core/`)
- ❌ Conceptos teóricos puros (van a `20_Concepts/`)

### Ejemplo:
- `NATIVE_ENGINE_COMMUNICATION.md` - Cómo funciona el motor nativo ✅
- `WORLD_DATA_TRANSFER_OPTIMIZATION.md` - Optimizaciones técnicas ✅
- `GPU_OPTIMIZATION.md` - Optimizaciones de GPU ✅
- `HISTORY_SYSTEM.md` - Sistema de historia ✅

---

## 📁 40_Experiments/ - Experimentos y Resultados

**Responsabilidad**: Resultados de experimentos, benchmarks, pruebas y guías prácticas.

### Qué incluir:
- ✅ Resultados de benchmarks
- ✅ Experimentos específicos (EXP_XXX)
- ✅ Guías de cómo ejecutar/pruebas
- ✅ Resultados de optimizaciones
- ✅ Comparaciones entre versiones
- ✅ Bitácoras de desarrollo
- ✅ Tests de visualizaciones
- ✅ Estudios de rendimiento real

### Qué NO incluir:
- ❌ Documentación técnica de componentes (van a `30_Components/`)
- ❌ Arquitectura general (van a `10_core/`)
- ❌ Conceptos teóricos (van a `20_Concepts/`)

### Ejemplo:
- `EXP_005_CPP_NATIVE_VICTORY.md` - Resultados del motor nativo ✅
- `BENCHMARK_TENSOR_STORAGE.md` - Benchmark de almacenamiento ✅
- `EXP_006_DATA_TRANSFER_OPTIMIZATION.md` - Experimentos de optimización ✅
- `HOW_TO_RUN.md` - Guía práctica de ejecución ✅
- `VISUALIZATION_TESTING.md` - Tests de visualización ✅

---

## 📁 00_Inbox/ - Pendientes de Clasificar

**Responsabilidad**: Documentos temporales que necesitan ser clasificados o eliminados.

### Qué incluir:
- ⏳ Notas temporales
- ⏳ Documentos pendientes de revisión
- ⏳ Borradores que necesitan ubicación

### Qué NO incluir:
- ❌ Documentación finalizada (debe moverse a su carpeta correspondiente)
- ❌ Documentos importantes (deben estar en su lugar final)

---

## 📁 99_Templates/ - Plantillas

**Responsabilidad**: Plantillas para crear nueva documentación.

### Qué incluir:
- ✅ Plantillas para componentes
- ✅ Guías de estilo
- ✅ Comandos de agentes
- ✅ Formatos estándar

---

## 🔀 Casos Específicos

### Optimizaciones

**Componente técnico** → `30_Components/`:
- Cómo funciona la optimización
- Arquitectura de la solución
- APIs y métodos

**Experimento/benchmark** → `40_Experiments/`:
- Resultados de pruebas
- Comparaciones de rendimiento
- Métricas reales

**Ejemplo**:
- `30_Components/WORLD_DATA_TRANSFER_OPTIMIZATION.md` - Cómo funciona ✅
- `40_Experiments/EXP_006_DATA_TRANSFER_OPTIMIZATION.md` - Resultados ✅

### Benchmarks

**Siempre** → `40_Experiments/`:
- Todos los benchmarks son experimentos
- Incluyen resultados y métricas

**Ejemplo**:
- `BENCHMARK_TENSOR_STORAGE.md` ✅
- `EXP_004_BENCHMARK_CPP_V1.md` ✅

### Guías de Uso

**Guía práctica/ejecución** → `40_Experiments/`:
- Cómo ejecutar
- Cómo probar
- Cómo usar

**Guía de aprendizaje** → `10_core/`:
- Progresión estructurada
- Niveles de aprendizaje

**Ejemplo**:
- `40_Experiments/HOW_TO_RUN.md` - Ejecución práctica ✅
- `10_core/PROGRESSIVE_LEARNING.md` - Aprendizaje estructurado ✅

---

## 📌 Resumen Visual

```
10_core/     → "¿Qué es el proyecto? ¿Cómo está diseñado?"
20_Concepts/ → "¿Qué conceptos teóricos fundamenta?"
30_Components/ → "¿Cómo funciona cada componente técnico?"
40_Experiments/ → "¿Qué resultados obtuvimos? ¿Cómo probar?"
00_Inbox/    → "Pendiente de clasificar"
99_Templates/ → "Plantillas para crear docs"
```

---

**Última actualización**: 2024-11-19

