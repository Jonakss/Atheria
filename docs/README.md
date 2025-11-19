# Atheria 4 - Documentación Completa

> **Vault de Obsidian** | Documentación técnica y conceptual del simulador de cosmogénesis

---

## 🗺️ Mapa de Contenidos (MOC)

### 📘 [📋 Core Documentation](10_core/00_CORE_MOC.md)
- **[Brief Maestro](10_core/ATHERIA_4_MASTER_BRIEF.md)** - Visión y objetivos del proyecto
- **[Arquitectura Técnica V4](10_core/TECHNICAL_ARCHITECTURE_V4.md)** - Arquitectura del sistema
- **[Glosario](10_core/ATHERIA_GLOSSARY.md)** - Terminología y conceptos clave
- **[Roadmap Fase 1](10_core/ROADMAP_PHASE_1.md)** - Plan de desarrollo inicial
- **[Roadmap Fase 2](10_core/ROADMAP_PHASE_2.md)** - Plan de desarrollo avanzado
- **[Aprendizaje Progresivo](10_core/PROGRESSIVE_LEARNING.md)** - Guía de aprendizaje estructurada
- **[Arquitectura de Inferencia Masiva](10_core/MASSIVE_INFERENCE_ARCHITECTURE.md)** - Escalabilidad horizontal

### 🧩 [🔧 Componentes Técnicos](30_Components/00_COMPONENTS_MOC.md)
- **[Modelos](30_Components/Models.md)** - Arquitecturas de modelos
- **[UNet](30_Components/UNET.md)** - Documentación específica de UNet
- **[Arquitectura V3](30_Components/ARCHITECTURE_V3.md)** - Sistema V3
- **[Motor Nativo C++](30_Components/NATIVE_ENGINE_COMMUNICATION.md)** - Comunicación motor nativo
- **[Sistema de Historia](30_Components/HISTORY_SYSTEM.md)** - Gestión de historia
- **[Optimización GPU](30_Components/GPU_OPTIMIZATION.md)** - Optimizaciones GPU
- **[Optimización de Transferencia](30_Components/WORLD_DATA_TRANSFER_OPTIMIZATION.md)** - Transferencia optimizada
- **[Recomendaciones de Visualización](30_Components/VISUALIZATION_RECOMMENDATIONS.md)** - Guía de visualizaciones
- Y más componentes técnicos...

### 🧪 [📊 Experimentos y Resultados](40_Experiments/00_EXPERIMENTS_MOC.md)
- **[Log de Desarrollo AI](40_Experiments/AI_DEV_LOG.md)** - Bitácora de desarrollo
- **[Guía de Experimentación](40_Experiments/EXPERIMENTATION_GUIDE.md)** - Cómo experimentar
- **[Guía de Pruebas de Visualización](40_Experiments/VISUALIZATION_TESTING.md)** - Tests de visualizaciones
- **[Cómo Ejecutar](40_Experiments/HOW_TO_RUN.md)** - Instrucciones de ejecución
- **[Benchmarks de Rendimiento](40_Experiments/)** - Resultados de benchmarks
- Y más experimentos...

### 💡 [🔬 Conceptos](20_Concepts/00_CONCEPTS_MOC.md)
- **[Vacío Armónico](20_Concepts/HARMONIC_VACUUM_CONCEPT.md)** - Concepto físico fundamental

### 📝 [📋 Plantillas](99_Templates/AGENT_TOOLKIT.md)
- **[Toolkit de Agente](99_Templates/AGENT_TOOLKIT.md)** - Comandos para agentes
- **[Guía de Agente](99_Templates/AGENT_GUIDELINES.md)** - Directrices para agentes IA
- **[Plantilla de Componente](99_Templates/Component_Template.md)** - Template para documentar componentes

---

## 🚀 Inicio Rápido

### Para Principiantes
1. Lee [Brief Maestro](10_core/ATHERIA_4_MASTER_BRIEF.md) para entender la visión
2. Sigue [Aprendizaje Progresivo](10_core/PROGRESSIVE_LEARNING.md) - Nivel 1
3. Prueba [Guía de Pruebas de Visualización](40_Experiments/VISUALIZATION_TESTING.md)

### Para Desarrolladores
1. Estudia [Arquitectura Técnica V4](10_core/TECHNICAL_ARCHITECTURE_V4.md)
2. Revisa [Motor Nativo C++](30_Components/NATIVE_ENGINE_COMMUNICATION.md)
3. Consulta [Guía de Agente](99_Templates/AGENT_GUIDELINES.md)

### Para Experimentadores
1. Consulta [Guía de Experimentación](40_Experiments/EXPERIMENTATION_GUIDE.md)
2. Revisa [Cómo Ejecutar](40_Experiments/HOW_TO_RUN.md)
3. Estudia [Log de Desarrollo AI](40_Experiments/AI_DEV_LOG.md)

---

## 📖 Estructura del Vault

```
docs/
├── README.md (este archivo)
├── 00_Inbox/                    # Notas pendientes de clasificar
│   └── notes_riscv.md
├── 10_core/                     # Documentación core
│   ├── 00_CORE_MOC.md
│   ├── ATHERIA_4_MASTER_BRIEF.md
│   ├── ATHERIA_GLOSSARY.md
│   ├── TECHNICAL_ARCHITECTURE_V4.md
│   ├── ROADMAP_PHASE_1.md
│   ├── ROADMAP_PHASE_2.md
│   ├── PROGRESSIVE_LEARNING.md
│   └── MASSIVE_INFERENCE_ARCHITECTURE.md
├── 20_Concepts/                 # Conceptos y teorías
│   ├── 00_CONCEPTS_MOC.md
│   └── HARMONIC_VACUUM_CONCEPT.md
├── 30_Components/               # Componentes técnicos
│   ├── 00_COMPONENTS_MOC.md
│   ├── Models.md
│   ├── UNET.md
│   ├── ARCHITECTURE_V3.md
│   ├── NATIVE_ENGINE_COMMUNICATION.md
│   ├── HISTORY_SYSTEM.md
│   ├── GPU_OPTIMIZATION.md
│   ├── WORLD_DATA_TRANSFER_OPTIMIZATION.md
│   └── ... (más componentes)
├── 40_Experiments/              # Experimentos y resultados
│   ├── 00_EXPERIMENTS_MOC.md
│   ├── AI_DEV_LOG.md
│   ├── EXPERIMENTATION_GUIDE.md
│   ├── VISUALIZATION_TESTING.md
│   ├── HOW_TO_RUN.md
│   └── ... (más experimentos)
└── 99_Templates/                # Plantillas
    ├── AGENT_TOOLKIT.md
    ├── AGENT_GUIDELINES.md
    └── Component_Template.md
```

---

## 🔗 Convenciones de Naming

- **Archivos:** `UPPERCASE_WITH_UNDERSCORES.md`
- **Carpetas:** `NN_Name/` (prefijo numérico para orden)
- **MOCs:** `00_CATEGORY_MOC.md` (Map of Content por categoría)
- **Enlaces:** Usar rutas relativas con nombres de archivo exactos

---

## 📋 Responsabilidades

Consulta **[Responsabilidades de Cada Carpeta](00_RESPONSIBILITIES.md)** para entender qué tipo de documentación va en cada carpeta.

---

## 📌 Tags para Obsidian

Usa estos tags para organizar:

- `#core` - Documentación core del proyecto
- `#component` - Componentes técnicos
- `#experiment` - Experimentos y resultados
- `#concept` - Conceptos teóricos
- `#guide` - Guías y tutoriales
- `#template` - Plantillas
- `#moc` - Map of Content

---

## 🎯 Próximos Pasos

1. **Explora los MOCs** de cada categoría
2. **Lee el Brief Maestro** para entender la visión
3. **Sigue la Guía de Aprendizaje Progresivo**
4. **Experimenta** siguiendo las guías de experimentación

---

*Última actualización: 2024-11-19*
