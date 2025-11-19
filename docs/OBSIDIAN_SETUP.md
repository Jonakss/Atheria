# 🔗 Configuración de Obsidian para Atheria 4

> Guía para configurar Obsidian como vault RAG y aprovechar todas sus características

---

## ✅ Requisitos Previos

1. **Instalar Obsidian**: [https://obsidian.md/](https://obsidian.md/)
2. **Abrir `docs/` como vault**: File → Open Folder → Seleccionar `docs/`
3. **Habilitar plugins esenciales**:
   - Graph View (incluido por defecto)
   - Backlinks (incluido por defecto)
   - Tag Pane (incluido por defecto)
   - Dataview (opcional pero recomendado para RAG)

---

## 🔗 Sistema de Enlaces Obsidian

### Formato Correcto

**✅ Usar formato Obsidian**:
```markdown
[[Archivo]]                    # Enlace simple
[[Carpeta/Archivo]]           # Enlace con ruta
[[Archivo|Texto Visible]]     # Enlace con alias
```

**❌ Evitar formato Markdown estándar**:
```markdown
[Texto](archivo.md)           # No funciona bien con backlinks
```

### Reglas de Naming

1. **Archivos**: `UPPERCASE_WITH_UNDERSCORES.md`
   - ✅ `SPATIAL_INDEXING.md`
   - ✅ `NATIVE_ENGINE_COMMUNICATION.md`
   - ❌ `spatial indexing.md` (espacios)
   - ❌ `SpatialIndexing.md` (camelCase)

2. **Caracteres especiales**: Evitar en nombres de archivo
   - ✅ `EXP_007_SPATIAL_INDEXING.md`
   - ❌ `EXP-007-Spatial-Indexing.md` (guiones pueden causar problemas)

3. **Consistencia**: Usar el mismo formato en enlaces y archivos

---

## 📋 Frontmatter YAML (Metadatos)

Cada archivo debe incluir metadatos YAML al inicio:

```yaml
---
title: Título del Documento
type: component | experiment | concept | guide | moc
status: active | deprecated | draft
tags: [tag1, tag2, tag3]
created: 2024-11-19
updated: 2024-11-19
aliases: [Alias 1, Alias 2]
related: [[Archivo1]], [[Archivo2]]
---
```

### Campos Importantes para RAG

- **`type`**: Categoría del documento (component, experiment, concept, etc.)
- **`tags`**: Tags para filtrado y búsqueda
- **`aliases`**: Nombres alternativos (útil para búsqueda)
- **`related`**: Enlaces relacionados explícitos

---

## 🏷️ Sistema de Tags

### Tags Principales

- `#core` - Documentación core del proyecto
- `#component` - Componentes técnicos
- `#experiment` - Experimentos y resultados
- `#concept` - Conceptos teóricos
- `#guide` - Guías y tutoriales
- `#template` - Plantillas
- `#moc` - Map of Content
- `#benchmark` - Resultados de benchmarks
- `#cpp` - Código C++
- `#native` - Motor nativo
- `#frontend` - Frontend y UI
- `#optimization` - Optimizaciones
- `#spatial` - Indexación espacial
- `#physics` - Conceptos físicos

### Tags Secundarios

- `#draft` - Borrador
- `#active` - Activo
- `#deprecated` - Deprecado
- `#verified` - Verificado
- `#todo` - Por hacer

---

## 🔗 Backlinks y Graph View

### Cómo Funcionan los Backlinks

Cuando enlazas `[[SPATIAL_INDEXING]]` desde otro archivo:
- **Enlace directo**: El archivo de origen aparece en "Linked mentions"
- **Backlink**: El archivo destino muestra "Backlinks"
- **Graph**: Aparece conexión en el grafo

### Ejemplo

**En `NATIVE_ENGINE_COMMUNICATION.md`**:
```markdown
Ver [[SPATIAL_INDEXING]] para optimización espacial.
```

**En `SPATIAL_INDEXING.md`**:
- **Backlinks pane** mostrará: `NATIVE_ENGINE_COMMUNICATION`
- **Graph view** mostrará conexión entre ambos

---

## 📊 MOCs (Maps of Content)

Los MOCs son índices navegables que conectan documentos relacionados:

- `00_CORE_MOC.md` - Documentación core
- `00_COMPONENTS_MOC.md` - Componentes técnicos
- `00_EXPERIMENTS_MOC.md` - Experimentos
- `00_CONCEPTS_MOC.md` - Conceptos

**Estructura de MOC**:
```markdown
# Título del MOC

## Categoría 1
- [[Archivo1]] - Descripción
- [[Archivo2]] - Descripción

## Categoría 2
- [[Archivo3]] - Descripción

## 🔗 Enlaces Relacionados
- [[Otro_MOC]] - Descripción
```

---

## 🔍 Uso como RAG

### Configuración de Plugins para RAG

1. **Dataview** (Recomendado)
   ```javascript
   // Listar todos los documentos de un tipo
   TABLE title, status, tags
   FROM #component
   WHERE status = "active"
   ```

2. **Omnisearch** (Recomendado)
   - Búsqueda semántica mejorada
   - Indexación de contenido

3. **Smart Random Note**
   - Útil para exploración aleatoria
   - Ayuda a descubrir conexiones

### Búsqueda Efectiva

- **Tags**: `tag:#component` para filtrar por tag
- **Backlinks**: Ver qué documentos enlazan a uno específico
- **Graph View**: Visualizar conexiones entre documentos
- **Dataview queries**: Consultas estructuradas

---

## ✅ Checklist de Validación

Para cada documento nuevo:

- [ ] Frontmatter YAML completo
- [ ] Tags apropiados
- [ ] Enlaces en formato `[[Archivo]]`
- [ ] Aliases si hay nombres alternativos
- [ ] Enlaces relacionados en frontmatter
- [ ] Referencia en MOC apropiado
- [ ] Nombres de archivo consistentes (UPPERCASE_WITH_UNDERSCORES)

---

## 🚀 Scripts Útiles

### Verificar Enlaces Rotos

En Obsidian: Settings → Files & Links → Automatically update internal links

### Generar Graph View

1. Abrir Graph View (Ctrl+G)
2. Configurar:
   - Show attachments: OFF
   - Show orphans: ON (para encontrar documentos sin enlaces)
   - Show tags: ON (opcional)

---

## 📝 Ejemplos

### Documento de Componente

```yaml
---
title: Spatial Indexing (Morton Codes)
type: component
status: active
tags: [component, optimization, spatial, cpp, verified]
created: 2024-11-19
updated: 2024-11-19
aliases: [Morton Codes, Z-order Curve, Spatial Optimization]
related: [[NATIVE_ENGINE_COMMUNICATION]], [[SPARSE_ENGINE]]
---

# Optimización Espacial (Spatial Indexing)

Ver también [[NATIVE_ENGINE_COMMUNICATION]] para integración.

[[00_COMPONENTS_MOC|← Volver al MOC de Componentes]]
```

### Documento de Experimento

```yaml
---
title: EXP_007 - Verificación de Spatial Indexing
type: experiment
status: completed
tags: [experiment, verification, spatial, qa]
created: 2024-11-19
related: [[SPATIAL_INDEXING]], [[EXP_006_DATA_TRANSFER_OPTIMIZATION]]
---

# EXP_007: Verificación de Spatial Indexing

Componente probado: [[SPATIAL_INDEXING]]

[[00_EXPERIMENTS_MOC|← Volver al MOC de Experimentos]]
```

---

## 🔗 Referencias

- [Obsidian Help - Links](https://help.obsidian.md/How+to/Internal+links)
- [Obsidian Help - Backlinks](https://help.obsidian.md/Plugins/Backlinks)
- [Obsidian Help - Graph View](https://help.obsidian.md/Plugins/Graph+view)
- [Dataview Plugin](https://blacksmithgu.github.io/obsidian-dataview/)

---

**Última actualización**: 2024-11-19

