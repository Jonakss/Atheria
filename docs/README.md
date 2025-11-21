---
title: Atheria 4 - Documentación Completa
type: index
status: active
tags: [core, documentation, index]
created: 2024-11-19
updated: 2025-11-20
aliases: [Documentation Index, Main Documentation]
---

# Atheria 4 - Documentación Completa

> **Vault de Obsidian** | Documentación técnica y conceptual del simulador de cosmogénesis

---

## 🗺️ Mapa de Contenidos (MOC)

### 📘 [[10_core/00_CORE_MOC|Core Documentation]]
- [[10_core/ATHERIA_4_MASTER_BRIEF|Brief Maestro]] - Visión y objetivos del proyecto
- [[10_core/TECHNICAL_ARCHITECTURE_V4|Arquitectura Técnica V4]] - Arquitectura del sistema
- [[10_core/ATHERIA_GLOSSARY|Glosario]] - Terminología y conceptos clave
- [[10_core/ROADMAP_PHASE_1|Roadmap Fase 1]] - Plan de desarrollo inicial
- [[10_core/ROADMAP_PHASE_2|Roadmap Fase 2]] - Plan de desarrollo avanzado
- [[10_core/PROGRESSIVE_LEARNING|Aprendizaje Progresivo]] - Guía de aprendizaje estructurada
- [[10_core/MASSIVE_INFERENCE_ARCHITECTURE|Arquitectura de Inferencia Masiva]] - Escalabilidad horizontal

### 🧩 [[30_Components/00_COMPONENTS_MOC|Componentes Técnicos]]
- [[30_Components/CLI_TOOL|CLI Tool]] - Herramienta de línea de comandos (atheria/ath)
- [[30_Components/Native_Engine_Core|Motor Nativo C++]] - Motor de alto rendimiento (LibTorch)
- [[30_Components/WEB_SOCKET_PROTOCOL|Protocolo WebSocket]] - Protocolo binario (MessagePack) vs JSON
- [[30_Components/Models|Modelos]] - Arquitecturas de modelos
- [[30_Components/UNET|UNet]] - Documentación específica de UNet
- [[30_Components/ARCHITECTURE_V3|Arquitectura V3]] - Sistema V3
- [[30_Components/NATIVE_ENGINE_COMMUNICATION|Motor Nativo C++ - Comunicación]] - Comunicación motor nativo
- [[30_Components/HISTORY_SYSTEM|Sistema de Historia]] - Gestión de historia
- [[30_Components/GPU_OPTIMIZATION|Optimización GPU]] - Optimizaciones GPU
- [[30_Components/WORLD_DATA_TRANSFER_OPTIMIZATION|Optimización de Transferencia]] - Transferencia optimizada
- [[30_Components/SPATIAL_INDEXING|Optimización Espacial]] - Índices espaciales con Morton Codes
- [[30_Components/VISUALIZATION_RECOMMENDATIONS|Recomendaciones de Visualización]] - Guía de visualizaciones
- Y más componentes técnicos...

### 🧪 [[40_Experiments/00_EXPERIMENTS_MOC|Experimentos y Resultados]]
- [[40_Experiments/HOW_TO_RUN|Cómo Ejecutar]] - Instrucciones completas con CLI (⭐ Empezar aquí)
- [[40_Experiments/AI_DEV_LOG|Log de Desarrollo AI]] - Bitácora de desarrollo y cambios recientes
- [[40_Experiments/EXPERIMENTATION_GUIDE|Guía de Experimentación]] - Cómo experimentar
- [[40_Experiments/VISUALIZATION_TESTING|Guía de Pruebas de Visualización]] - Tests de visualizaciones
- [[40_Experiments/NATIVE_ENGINE_PERFORMANCE_ISSUES|Problemas de Rendimiento Motor Nativo]] - Troubleshooting
- [[40_Experiments/BENCHMARK_TENSOR_STORAGE|Benchmarks de Rendimiento]] - Resultados de benchmarks
- Y más experimentos...

### 💡 [[20_Concepts/00_CONCEPTS_MOC|Conceptos]]
- [[20_Concepts/HARMONIC_VACUUM_CONCEPT|Vacío Armónico]] - Concepto físico fundamental

### 📝 [[99_Templates/AGENT_TOOLKIT|Plantillas]]
- [[99_Templates/AGENT_TOOLKIT|Toolkit de Agente]] - Comandos para agentes
- [[99_Templates/AGENT_GUIDELINES|Guía de Agente]] - Directrices para agentes IA
- [[99_Templates/Component_Template|Plantilla de Componente]] - Template para documentar componentes

---

## 🚀 Inicio Rápido

### Para Principiantes
1. Lee [[10_core/ATHERIA_4_MASTER_BRIEF|Brief Maestro]] para entender la visión
2. Sigue [[10_core/PROGRESSIVE_LEARNING|Aprendizaje Progresivo]] - Nivel 1
3. Prueba [[40_Experiments/VISUALIZATION_TESTING|Guía de Pruebas de Visualización]]

### Para Desarrolladores
1. Lee [[40_Experiments/HOW_TO_RUN|Cómo Ejecutar]] - Instrucciones de instalación y CLI
2. Estudia [[10_core/TECHNICAL_ARCHITECTURE_V4|Arquitectura Técnica V4]]
3. Revisa [[30_Components/Native_Engine_Core|Motor Nativo C++]]
4. Consulta [[30_Components/CLI_TOOL|CLI Tool]] para desarrollo
5. Consulta [[99_Templates/AGENT_GUIDELINES|Guía de Agente]]

### Para Experimentadores
1. Consulta [[40_Experiments/EXPERIMENTATION_GUIDE|Guía de Experimentación]]
2. Revisa [[40_Experiments/HOW_TO_RUN|Cómo Ejecutar]]
3. Estudia [[40_Experiments/AI_DEV_LOG|Log de Desarrollo AI]]

---

## 📖 Estructura del Vault

```
docs/
├── README.md (este archivo)
├── OBSIDIAN_SETUP.md        # 🔗 Guía de configuración de Obsidian
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
│   ├── SPATIAL_INDEXING.md
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
- **Enlaces:** Usar formato Obsidian `[[archivo]]` o `[[carpeta/archivo]]`

---

## 📋 Configuración de Obsidian

Para usar este vault como sistema RAG y aprovechar todas las características de Obsidian:

👉 **Ver [[OBSIDIAN_SETUP|Guía de Configuración de Obsidian]]**

Incluye:
- ✅ Configuración de enlaces y backlinks
- ✅ Frontmatter YAML para metadatos
- ✅ Sistema de tags
- ✅ Uso de Graph View
- ✅ Configuración de plugins para RAG

---

## 📋 Responsabilidades

Consulta [[00_RESPONSIBILITIES|Responsabilidades de Cada Carpeta]] para entender qué tipo de documentación va en cada carpeta.

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

*Última actualización: 2025-11-20*
