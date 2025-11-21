# Knowledge Base - Atheria 4

**IMPORTANTE:** Este documento explica cómo funciona la **BASE DE CONOCIMIENTOS** de Atheria 4.

## ¿Qué es la Knowledge Base?

La carpeta `docs/` **NO es solo documentación del proyecto**. Es la **BASE DE CONOCIMIENTOS (Knowledge Base)** que los agentes de IA consultan para:

1. **Entender el contexto histórico** - ¿Por qué se tomó una decisión?
2. **Conocer decisiones de diseño** - ¿Qué alternativas se consideraron?
3. **Reutilizar conocimiento** - ¿Cómo se resolvió un problema similar antes?
4. **Mantener consistencia** - ¿Qué patrones y convenciones existen?
5. **Evitar reinventar la rueda** - ¿Qué ya se implementó?

## Estructura de la Knowledge Base

```
docs/
├── 10_Core/              # Conocimiento fundamental (MASTER_BRIEF, arquitectura, glosario)
├── 20_Concepts/          # Conceptos y teorías
├── 30_Components/        # Componentes técnicos y sus implementaciones
├── 40_Experiments/       # Experimentos, resultados, lecciones aprendidas
├── 99_Templates/         # Plantillas para documentación consistente
└── 00_*_MOC.md          # Mapas de contenido (navegación)
```

## Principios RAG (Retrieval-Augmented Generation)

### 1. Consultar ANTES de Implementar

**❌ INCORRECTO:**
```
Agente: "Voy a implementar X"
[Implementa X sin consultar docs/]
```

**✅ CORRECTO:**
```
Agente: "Voy a implementar X"
[Consulta docs/30_Components/ para ver si existe algo similar]
[Consulta docs/40_Experiments/ para ver experimentos anteriores]
[Consulta docs/20_Concepts/ para entender conceptos relacionados]
[Implementa X usando el conocimiento existente]
[Actualiza docs/ con nueva información]
```

### 2. Documentar POR QUÉ, no solo QUÉ

La knowledge base debe explicar **razones** y **decisiones**, no solo hechos.

**❌ Documentación pobre (solo QUÉ):**
```markdown
## Feature X
Se implementó usando React y TypeScript.
```

**✅ Knowledge base rica (QUÉ + POR QUÉ):**
```markdown
## Feature X

### Decisión de Implementación
Se implementó usando React y TypeScript.

### Alternativas Consideradas
1. **Vue.js** - Rechazado porque el equipo tiene más experiencia con React
2. **Vanilla JS** - Rechazado porque la complejidad requiere un framework

### Trade-offs
- ✅ **Pros:** Consistencia con frontend existente, type safety
- ❌ **Contras:** Bundle size mayor, curva de aprendizaje inicial

### Problemas que Resuelve
- Problema Y: Solución Z
- Problema A: Solución B

### Referencias
- [[TECHNICAL_ARCHITECTURE_V4.md]] - Arquitectura general
- [[Component_Template.md]] - Patrón de componentes
```

### 3. Enlaces entre Conceptos

Usa enlaces `[[archivo]]` para conectar conceptos relacionados. Esto ayuda a los agentes a:
- Descubrir información relacionada
- Entender dependencias
- Navegar la knowledge base efectivamente

**Ejemplo:**
```markdown
El motor nativo usa [[lazy_conversion]] para optimizar rendimiento.
Ver [[VISUALIZATION_OPTIMIZATION_ANALYSIS.md]] para más detalles.
```

### 4. Actualizar al Cambiar

**Regla de oro:** Si cambias código que afecta conocimiento documentado, actualiza la knowledge base.

**Flujo correcto:**
1. Consultar `docs/` para entender contexto
2. Implementar cambio
3. Actualizar `docs/` con nueva información
4. Commit de código + documentación juntos

## Tipos de Conocimiento

### 1. Conocimiento Fundamental (`10_Core/`)
- Visión del proyecto
- Arquitectura técnica
- Glosario de términos
- Roadmaps

**Cuándo consultar:** Antes de cualquier cambio significativo.

### 2. Conceptos (`20_Concepts/`)
- Teorías y modelos
- Conceptos abstractos
- Relaciones entre conceptos

**Cuándo consultar:** Al trabajar con nueva funcionalidad o conceptos complejos.

### 3. Componentes (`30_Components/`)
- Implementaciones técnicas
- APIs y interfaces
- Patrones de código
- Decisiones de implementación

**Cuándo consultar:** Al implementar o modificar componentes.

### 4. Experimentos (`40_Experiments/`)
- Pruebas y resultados
- Lecciones aprendidas
- Métricas y benchmarks
- Decisiones basadas en evidencia

**Cuándo consultar:** Antes de experimentar, para ver qué ya se probó.

## Mapas de Contenido (MOC)

Los archivos `00_*_MOC.md` son mapas de contenido que facilitan la navegación:

- `00_COMPONENTS_MOC.md` - Índice de componentes
- `00_CONCEPTS_MOC.md` - Índice de conceptos
- `00_EXPERIMENTS_MOC.md` - Índice de experimentos

**Regla:** Actualizar MOC cuando se agreguen nuevos documentos.

## Plantillas

`docs/99_Templates/` contiene plantillas para documentación consistente:

- `Component_Template.md` - Para documentar componentes
- `Experiment_Template.md` - Para documentar experimentos
- `Concept_Template.md` - Para documentar conceptos

**Uso:** Usar plantillas para mantener consistencia y facilitar RAG.

## Mejores Prácticas

### ✅ HACER

1. **Consultar primero:** Siempre buscar en `docs/` antes de implementar
2. **Explicar por qué:** Documentar razones y decisiones, no solo hechos
3. **Enlazar conceptos:** Usar `[[archivo]]` para conectar ideas relacionadas
4. **Actualizar MOC:** Mantener índices actualizados
5. **Commit juntos:** Cambios de código + documentación en mismo commit

### ❌ NO HACER

1. **No consultar:** Implementar sin revisar conocimiento existente
2. **Solo documentar qué:** No explicar razones o contexto
3. **Documentación huérfana:** Crear documentos sin enlaces
4. **MOC desactualizados:** Olvidar actualizar índices
5. **Commits separados:** Separar código de documentación

## Ejemplo de Uso Correcto

**Escenario:** Implementar nueva optimización de visualización

1. **Consultar knowledge base:**
   - `docs/30_Components/VISUALIZATION_OPTIMIZATION_ANALYSIS.md`
   - `docs/40_Experiments/` para ver optimizaciones anteriores
   - `docs/20_Concepts/` para conceptos de rendimiento

2. **Entender contexto:**
   - ¿Qué optimizaciones ya existen?
   - ¿Qué problemas resolvieron?
   - ¿Qué trade-offs tienen?

3. **Implementar:**
   - Usar conocimiento existente
   - Aplicar patrones establecidos
   - Mantener consistencia

4. **Documentar:**
   - Actualizar `VISUALIZATION_OPTIMIZATION_ANALYSIS.md`
   - Agregar entrada a `AI_DEV_LOG.md`
   - Actualizar MOC relevante
   - Incluir enlaces `[[archivo]]`

5. **Commit:**
   - Código + documentación juntos
   - Mensaje descriptivo con referencia a docs

## Referencias

- [[.cursorrules]] - Reglas para agentes
- [[OBSIDIAN_SETUP.md]] - Configuración de Obsidian
- [[ATHERIA_4_MASTER_BRIEF.md]] - Visión del proyecto
- [[TECHNICAL_ARCHITECTURE_V4.md]] - Arquitectura técnica

---

**Última actualización:** 2025-01-21  
**Mantenido por:** Agentes de IA y desarrolladores

