# Documentación de Aetheria

Bienvenido a la documentación completa de Aetheria Simulation Lab.

## 📚 Índice de Documentación

### Guías Principales

1. **[Guía de Aprendizaje Progresivo](PROGRESSIVE_LEARNING.md)**
   - Aprende desde lo básico hasta experimentos avanzados
   - Roadmap de aprendizaje por semanas
   - Ejercicios prácticos

2. **[Guía de Experimentación](EXPERIMENTATION_GUIDE.md)**
   - Estrategias de experimentación
   - Cómo probar cada visualización
   - Optimizaciones y eficiencias
   - Ejemplos de experimentos

3. **[Guía de Pruebas por Visualización](VISUALIZATION_TESTING.md)**
   - Cómo probar cada visualización
   - Qué buscar en cada una
   - Interpretación de resultados
   - Combinaciones útiles

4. **[Recomendaciones de Visualizaciones](VISUALIZATION_RECOMMENDATIONS.md)**
   - Análisis de visualizaciones disponibles
   - Prioridades de implementación
   - Costos y beneficios

5. **[Análisis de Técnicas Avanzadas](TECHNIQUES_ANALYSIS.md)**
   - RMSNorm, SwiGLU, RoPE
   - Cuándo usar cada técnica
   - Implementación y optimización

6. **[Zoom y Transferencia de Datos](ZOOM_AND_DATA_TRANSFER.md)**
   - Cómo funciona el zoom actual
   - Optimizaciones disponibles (downsampling)
   - Recomendaciones de uso

7. **[Arquitectura para Inferencia Masiva](MASSIVE_INFERENCE_ARCHITECTURE.md)**
   - Clustering y distribución de simulaciones
   - Protocolos de comunicación entre workers
   - Escalabilidad horizontal
   - Casos de uso para búsqueda masiva de patrones

---

## 🚀 Inicio Rápido

### Para Principiantes
1. Lee [Guía de Aprendizaje Progresivo](PROGRESSIVE_LEARNING.md) - Nivel 1
2. Prueba [Guía de Pruebas por Visualización](VISUALIZATION_TESTING.md) - Visualizaciones Básicas
3. Sigue los experimentos del [Nivel 1](PROGRESSIVE_LEARNING.md#nivel-1-fundamentos-semanas-1-2)

### Para Usuarios Avanzados
1. Revisa [Guía de Experimentación](EXPERIMENTATION_GUIDE.md)
2. Consulta [Recomendaciones de Visualizaciones](VISUALIZATION_RECOMMENDATIONS.md)
3. Implementa técnicas de [Análisis de Técnicas Avanzadas](TECHNIQUES_ANALYSIS.md)

### Para Desarrolladores
1. Estudia [Arquitectura para Inferencia Masiva](MASSIVE_INFERENCE_ARCHITECTURE.md)
2. Planifica escalabilidad y clustering
3. Implementa protocolos de comunicación distribuida

---

## 📖 Estructura de Documentación

```
docs/
├── README.md (este archivo)
├── 00_Inbox/                    # Notas y documentos pendientes de clasificar
├── 10_core/                     # Documentación core del proyecto
│   ├── ATHERIA_4_MASTER_BRIEF.md
│   ├── ATHERIA_GLOSSARY.md
│   ├── ROADMAP_PHASE_1.md
│   ├── TECHNICAL_ARCHITECTURE_V4.md
│   └── PROGRESSIVE_LEARNING.md
├── 20_Concepts/                 # Conceptos y teorías
│   └── Concepto_Vacio_Armonico.md
├── 30_Components/               # Documentación de componentes técnicos
│   ├── Models.md                # Arquitecturas de modelos
│   ├── ArchitectureV3.md        # Arquitectura del sistema (V3)
│   └── HISTORY_SYSTEM.md        # Sistema de historia
├── 40_Experiments/              # Experimentos y resultados
│   ├── AI_DEV_LOG.md
│   └── Progressive_Training.md
├── 99_Templates/                # Plantillas para documentación
│   └── Component_Template.md
├── EXPERIMENTATION_GUIDE.md     # Cómo experimentar
├── VISUALIZATION_TESTING.md     # Pruebas por visualización
├── VISUALIZATION_RECOMMENDATIONS.md  # Análisis de visualizaciones
├── TECHNIQUES_ANALYSIS.md       # Técnicas avanzadas (RMSNorm, RoPE, etc.)
├── ZOOM_AND_DATA_TRANSFER.md    # Optimización de zoom y datos
└── MASSIVE_INFERENCE_ARCHITECTURE.md  # Arquitectura para inferencia masiva
```

---

## 🎯 Objetivos de Aprendizaje

### Nivel 1: Fundamentos
- ✅ Entender física básica (QCA, unitariedad, Lindblad)
- ✅ Dominar visualizaciones básicas (density, phase, energy)
- ✅ Comparar arquitecturas simples (MLP vs UNet)

### Nivel 2: Herramientas
- ✅ Dominar todas las visualizaciones
- ✅ Usar t-SNE para análisis
- ✅ Guardar y analizar historia

### Nivel 3: Optimización
- ✅ Encontrar mejores parámetros
- ✅ Optimizar para tu hardware
- ✅ Documentar configuraciones exitosas

### Nivel 4: A-Life
- ✅ Buscar gliders
- ✅ Buscar osciladores
- ✅ Buscar replicadores
- ✅ Caracterizar estructuras encontradas

---

## 🔧 Recursos Técnicos

### Comandos Útiles
```javascript
// Habilitar historia
simulation.enable_history({enabled: true})

// Guardar historia
simulation.save_history({filename: "experimento.json"})

// Capturar snapshot
simulation.capture_snapshot({})

// Configurar FPS
simulation.set_fps({fps: 30})

// Configurar velocidad
simulation.set_speed({speed: 2.0})
```

### Visualizaciones Disponibles
- **Básicas:** density, phase, energy, real, imag
- **Avanzadas:** entropy, coherence, channel_activity, physics
- **Análisis:** spectral, gradient, flow, phase_attractor
- **t-SNE:** universe_atlas, cell_chemistry

---

## 📝 Notas

- La documentación se actualiza constantemente
- Si encuentras errores o tienes sugerencias, documenta tus hallazgos
- Comparte configuraciones exitosas con la comunidad

---

## 🎓 Próximos Pasos

1. **Lee la [Guía de Aprendizaje Progresivo](PROGRESSIVE_LEARNING.md)**
2. **Prueba las visualizaciones** según [VISUALIZATION_TESTING.md](VISUALIZATION_TESTING.md)
3. **Experimenta** siguiendo [EXPERIMENTATION_GUIDE.md](EXPERIMENTATION_GUIDE.md)
4. **Busca A-Life** usando todas las herramientas

¡Buena suerte en tu búsqueda de vida artificial! 🚀

